import os
import glob
import torch
from torch.utils.data import DataLoader

from config import device, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from data.dataset import TrajectoryDataset
from models.generator import TrajectoryPredictor
from models.discriminator import TrajectoryDiscriminator
from training.trainer import GANTrainer


def load_data():
    """加载数据"""
    # 自动获取文件列表
    train_files = glob.glob(os.path.join(DATA_CONFIG['train_dir'], "*.txt"))
    val_files = glob.glob(os.path.join(DATA_CONFIG['val_dir'], "*.txt"))
    test_files = glob.glob(os.path.join(DATA_CONFIG['test_dir'], "*.txt"))

    train_files.sort()
    val_files.sort()
    test_files.sort()

    print(f"找到 {len(train_files)} 个训练文件")
    print(f"找到 {len(val_files)} 个验证文件")
    print(f"找到 {len(test_files)} 个测试文件")

    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError("错误：没有找到训练或验证数据文件！")

    # 创建数据集和数据加载器
    train_dataset = TrajectoryDataset(train_files, DATA_CONFIG['obs_len'],
                                      DATA_CONFIG['pred_len'], DATA_CONFIG['num_nodes'])
    val_dataset = TrajectoryDataset(val_files, DATA_CONFIG['obs_len'],
                                    DATA_CONFIG['pred_len'], DATA_CONFIG['num_nodes'])

    # 测试集
    test_dataset = None
    test_loader = None
    if len(test_files) > 0:
        test_dataset = TrajectoryDataset(test_files, DATA_CONFIG['obs_len'],
                                         DATA_CONFIG['pred_len'], DATA_CONFIG['num_nodes'])
        test_loader = DataLoader(test_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=False)
        print(f"测试集大小: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("错误：训练集或验证集为空！")

    train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 打印数据形状示例
    sample_obs, sample_pred = train_dataset[0]
    print(f"观察序列形状: {sample_obs.shape}")
    print(f"预测序列形状: {sample_pred.shape}")

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_models():
    """创建模型"""
    # 创建生成器和判别器
    generator = TrajectoryPredictor(
        obs_len=DATA_CONFIG['obs_len'],
        pred_len=DATA_CONFIG['pred_len'],
        num_nodes=DATA_CONFIG['num_nodes'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_stgat_blocks=MODEL_CONFIG['num_stgat_blocks'],
        num_transformer_layers=MODEL_CONFIG['num_transformer_layers']
    ).to(device)

    discriminator = TrajectoryDiscriminator(
        seq_len=DATA_CONFIG['seq_len'],
        num_nodes=DATA_CONFIG['num_nodes'],
        hidden_size=MODEL_CONFIG['hidden_size'] // 2
    ).to(device)

    # 打印参数数量对比
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())

    print(f"生成器参数: {g_params}")
    print(f"判别器参数: {d_params}")
    print(f"参数比例: {d_params / g_params * 100:.2f}%")

    return generator, discriminator


def test_models(generator, discriminator, train_loader):
    """测试模型前向传播"""
    print("测试判别器前向传播...")
    try:
        test_batch = next(iter(train_loader))
        test_obs, test_true = test_batch
        test_obs = test_obs.to(device)
        test_true = test_true.to(device)

        test_real_traj = torch.cat([test_obs, test_true], dim=1)
        test_output = discriminator(test_real_traj)
        print(f"判别器测试输出形状: {test_output.shape}")
        print("判别器前向传播测试通过!")
    except Exception as e:
        print(f"判别器测试失败: {e}")
        raise


def train_models(generator, discriminator, train_loader, val_loader):
    """训练模型"""
    # 定义优化器
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=1e-5)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=1e-5)

    # 创建训练器
    trainer = GANTrainer(generator, discriminator, train_loader, val_loader,
                         g_optimizer, d_optimizer, device, TRAIN_CONFIG)

    # 训练循环
    train_g_losses = []
    train_d_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        # 训练一个epoch
        avg_g_loss, avg_d_loss, avg_x_sum_loss, g_loss_adv, g_loss_mse, d_loss_real, d_loss_fake = trainer.train_epoch(
            epoch)

        # 验证
        avg_val_loss, avg_val_x_sum_loss = trainer.validate()

        # 记录损失
        train_g_losses.append(avg_g_loss)
        train_d_losses.append(avg_d_loss)
        val_losses.append(avg_val_loss)

        # 打印进度
        print(f'Epoch {epoch + 1}/{TRAIN_CONFIG["num_epochs"]}:')
        print(f'  G Loss: {avg_g_loss:.6f} (Adv: {g_loss_adv:.6f}, MSE: {g_loss_mse:.6f}, X_Sum: {avg_x_sum_loss:.6f})')
        print(f'  D Loss: {avg_d_loss:.6f} (Real: {d_loss_real:.6f}, Fake: {d_loss_fake:.6f})')
        print(f'  Val Loss: {avg_val_loss:.6f} (X_Sum: {avg_val_x_sum_loss:.6f})')

        # 保存检查点
        if (epoch + 1) % TRAIN_CONFIG['save_freq'] == 0 or epoch == 0:
            checkpoint_path = trainer.save_checkpoint(epoch, avg_g_loss, avg_d_loss, avg_val_loss, avg_val_x_sum_loss)
            print(f'保存中间模型: {checkpoint_path}')

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trainer.save_best_model(epoch, best_val_loss)
            print(f'保存最佳GAN模型，验证损失: {best_val_loss:.6f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                print(f'早停：验证损失在{TRAIN_CONFIG["patience"]}个epoch内没有改善')
                break

    # 保存最终模型
    final_model_path = trainer.save_final_model(epoch, train_g_losses, train_d_losses, val_losses)
    print(f'保存最终模型: {final_model_path}')

    return train_g_losses, train_d_losses, val_losses, best_val_loss

def main():
    """主函数"""
    print(f'Using device: {device}')

    try:
        # 加载数据
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = load_data()

        # 创建模型
        generator, discriminator = create_models()

        # 测试模型
        test_models(generator, discriminator, train_loader)

        # 训练模型
        train_g_losses, train_d_losses, val_losses, best_val_loss = train_models(
            generator, discriminator, train_loader, val_loader
        )

        # 加载最佳模型
        print(f"加载最佳GAN模型，验证损失: {best_val_loss:.6f}")
        checkpoint = torch.load(TRAIN_CONFIG['best_model_path'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    except Exception as e:
        print(f"程序执行出错: {e}")
        raise


if __name__ == "__main__":
    main()