import torch
import torch.nn as nn
from .components import STGATBlock, PositionalEncoding

class TrajectoryPredictor(nn.Module):
    def __init__(self, obs_len=60, pred_len=140, num_nodes=22,
                 input_features=2, hidden_size=64, num_heads=4,
                 num_stgat_blocks=2, num_transformer_layers=2, dropout=0.1):
        super(TrajectoryPredictor, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.input_embedding = nn.Linear(input_features, hidden_size)
        self.stgat_blocks = nn.ModuleList()
        for i in range(num_stgat_blocks):
            in_features = hidden_size if i == 0 else hidden_size
            self.stgat_blocks.append(
                STGATBlock(in_features, hidden_size, num_heads, dropout)
            )
        self.pos_encoder = PositionalEncoding(hidden_size, dropout, max_len=500)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_transformer_layers,
            num_decoder_layers=num_transformer_layers,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder_query = nn.Parameter(torch.randn(pred_len, hidden_size))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 2)
        )

    def forward(self, obs_traj):
        batch_size = obs_traj.size(0)

        x = self.input_embedding(obs_traj)
        for stgat_block in self.stgat_blocks:
            x = stgat_block(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size * self.num_nodes, self.obs_len, self.hidden_size)
        x = self.pos_encoder(x)
        decoder_input = self.decoder_query.unsqueeze(0).repeat(batch_size * self.num_nodes, 1, 1)

        memory = self.transformer.encoder(x)
        output = self.transformer.decoder(decoder_input, memory)

        output = self.output_layer(output)

        pred_traj = output.reshape(batch_size, self.num_nodes, self.pred_len, 2)
        pred_traj = pred_traj.permute(0, 2, 1, 3)

        return pred_traj