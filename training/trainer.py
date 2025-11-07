import torch
import torch.nn as nn
import os
from tqdm import tqdm


class GANTrainer:
    def __init__(self, generator, discriminator, train_loader, val_loader,
                 g_optimizer, d_optimizer, device, config):
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.config = config

        # 创建检查点目录
        if not os.path.exists(config['checkpoint_dir']):
            os.makedirs(config['checkpoint_dir'])

        # 损失函数
        self.adversarial_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.generator.train()
        self.discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_x_sum_loss = 0
        real_label = 1.0
        fake_label = 0.0

        for batch_idx, (obs_traj, true_traj) in enumerate(tqdm(self.train_loader,
                                                               desc=f"Epoch {epoch + 1}/{self.config['num_epochs']}")):
            batch_size = obs_traj.size(0)
            obs_traj = obs_traj.to(self.device)
            true_traj = true_traj.to(self.device)

            # 准备真实数据（观察+真实未来）
            real_trajectories = torch.cat([obs_traj, true_traj], dim=1)

            # 创建真实和假的标签
            real_labels = torch.full((batch_size, 1), real_label, device=self.device)
            fake_labels = torch.full((batch_size, 1), fake_label, device=self.device)
            # 训练判别器
            self.d_optimizer.zero_grad()

            # 真实数据的损失
            real_output = self.discriminator(real_trajectories)
            d_loss_real = self.adversarial_loss(real_output, real_labels)

            # 生成假数据
            with torch.no_grad():
                fake_future = self.generator(obs_traj)

            fake_trajectories = torch.cat([obs_traj, fake_future], dim=1)

            # 假数据的损失
            fake_output = self.discriminator(fake_trajectories.detach())
            d_loss_fake = self.adversarial_loss(fake_output, fake_labels)

            # 总判别器损失
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            self.d_optimizer.step()

            # 训练生成器

            self.g_optimizer.zero_grad()

            fake_future = self.generator(obs_traj)
            fake_trajectories = torch.cat([obs_traj, fake_future], dim=1)

            fake_output = self.discriminator(fake_trajectories)
            g_loss_adv = self.adversarial_loss(fake_output, real_labels)
            g_loss_mse = self.mse_loss(fake_future, true_traj)

            true_x_sum = torch.sum(true_traj[:, :, :, 0], dim=2)
            pred_x_sum = torch.sum(fake_future[:, :, :, 0], dim=2)
            g_loss_x_sum = self.mse_loss(pred_x_sum, true_x_sum)

            # 总生成器损失
            g_loss = g_loss_adv + 10 * g_loss_mse + self.config['x_sum_loss_weight'] * g_loss_x_sum
            g_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_x_sum_loss += g_loss_x_sum.item()

        avg_g_loss = epoch_g_loss / len(self.train_loader)
        avg_d_loss = epoch_d_loss / len(self.train_loader)
        avg_x_sum_loss = epoch_x_sum_loss / len(self.train_loader)

        return avg_g_loss, avg_d_loss, avg_x_sum_loss, g_loss_adv.item(), g_loss_mse.item(), d_loss_real.item(), d_loss_fake.item()

    def validate(self):
        self.generator.eval()
        self.discriminator.eval()
        epoch_val_loss = 0
        epoch_val_x_sum_loss = 0

        with torch.no_grad():
            for obs_traj, true_traj in self.val_loader:
                obs_traj, true_traj = obs_traj.to(self.device), true_traj.to(self.device)

                fake_future = self.generator(obs_traj)

                val_loss = self.mse_loss(fake_future, true_traj)
                epoch_val_loss += val_loss.item()

                true_x_sum = torch.sum(true_traj[:, :, :, 0], dim=2)
                pred_x_sum = torch.sum(fake_future[:, :, :, 0], dim=2)
                val_x_sum_loss = self.mse_loss(pred_x_sum, true_x_sum)
                epoch_val_x_sum_loss += val_x_sum_loss.item()

        avg_val_loss = epoch_val_loss / len(self.val_loader)
        avg_val_x_sum_loss = epoch_val_x_sum_loss / len(self.val_loader)

        return avg_val_loss, avg_val_x_sum_loss

    def save_checkpoint(self, epoch, g_loss, d_loss, val_loss, val_x_sum_loss):
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'train_g_loss': g_loss,
            'train_d_loss': d_loss,
            'val_loss': val_loss,
            'val_x_sum_loss': val_x_sum_loss
        }, checkpoint_path)
        return checkpoint_path

    def save_best_model(self, epoch, best_val_loss):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, self.config['best_model_path'])

    def save_final_model(self, epoch, train_g_losses, train_d_losses, val_losses):
        final_model_path = os.path.join(self.config['checkpoint_dir'], 'final_model.pth')
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'epoch': epoch,
            'train_g_losses': train_g_losses,
            'train_d_losses': train_d_losses,
            'val_losses': val_losses
        }, final_model_path)
        return final_model_path