import torch

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据参数
DATA_CONFIG = {
    'obs_len': 90,
    'pred_len': 450,
    'seq_len': 540,  # obs_len + pred_len
    'num_nodes': 22,
    'batch_size': 32,
    'train_dir': "data/train",
    'val_dir': "data/val",
    'test_dir': "data/test"
}

# 模型参数
MODEL_CONFIG = {
    'hidden_size': 32,
    'num_heads': 4,
    'num_stgat_blocks': 1,
    'num_transformer_layers': 1,
    'dropout': 0.1,
    'input_features': 2
}

# 训练参数
TRAIN_CONFIG = {
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'patience': 10,
    'x_sum_loss_weight': 0.1,
    'save_freq': 1,
    'checkpoint_dir': 'checkpoints',
    'best_model_path': 'best_gan_model.pth',
    'final_model_path': 'final_model.pth'
}