import numpy as np
import torch
from torch.utils.data import Dataset
import os

class TrajectoryDataset(Dataset):
    def __init__(self, file_paths, obs_len=60, pred_len=140, num_nodes=22):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.seq_len = obs_len + pred_len
        self.data = []
        self.means = []  # 保存每个序列的均值
        self.stds = []  # 保存每个序列的标准差
        self.original_data = []  # 保存原始数据

        for file_path in file_paths:
            try:
                data = np.loadtxt(file_path, delimiter='\t')
                if len(data) == 0:
                    continue

                frames = data[:, 0].astype(int)
                unique_frames = np.unique(frames)
                num_frames = len(unique_frames)

                if num_frames < self.seq_len:
                    continue
                traj_data = np.zeros((self.seq_len, self.num_nodes, 2))
                for i, frame in enumerate(unique_frames[:self.seq_len]):
                    frame_data = data[data[:, 0] == frame]
                    nodes = frame_data[:, 1].astype(int)
                    for node in nodes:
                        if node < self.num_nodes:
                            traj_data[i, node, :] = frame_data[frame_data[:, 1] == node, 2:4]
                self.original_data.append(traj_data.copy())
                normalized_data, mean, std = self.normalize_trajectory(traj_data)
                self.data.append(normalized_data)
                self.means.append(mean)
                self.stds.append(std)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

        print(f"Loaded {len(self.data)} sequences")

    def normalize_trajectory(self, traj_data):
        normalized_data = np.zeros_like(traj_data)
        means = np.zeros((self.num_nodes, 2))
        stds = np.zeros((self.num_nodes, 2))

        for node in range(traj_data.shape[1]):
            node_data = traj_data[:, node, :]
            if np.any(node_data != 0):
                mean = node_data.mean(axis=0)
                std = node_data.std(axis=0)
                std = np.where(std == 0, 1, std)  # 避免除零
                normalized_data[:, node, :] = (node_data - mean) / std
                means[node] = mean
                stds[node] = std
            else:
                means[node] = 0
                stds[node] = 1

        return normalized_data, means, stds

    def denormalize_trajectory(self, normalized_data, idx):
        mean = self.means[idx]
        std = self.stds[idx]
        return normalized_data * std + mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs_traj = self.data[idx][:self.obs_len]
        pred_traj = self.data[idx][self.obs_len:]
        return torch.FloatTensor(obs_traj), torch.FloatTensor(pred_traj)