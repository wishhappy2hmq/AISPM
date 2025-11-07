import torch
import torch.nn as nn

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, seq_len=200, num_nodes=22, input_features=2,
                 hidden_size=32, dropout=0.1):
        super(TrajectoryDiscriminator, self).__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.flatten_size = seq_len * num_nodes * input_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.flatten_size, hidden_size * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, traj):
        batch_size = traj.size(0)
        x = traj.reshape(batch_size, -1)

        features = self.feature_extractor(x)
        validity = self.classifier(features)

        return validity