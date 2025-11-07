import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h, adj_mask=None):
        batch_size, seq_len, num_nodes, _ = h.shape
        h_flat = h.reshape(batch_size * seq_len, num_nodes, -1)

        # 线性变换
        Wh = self.W(h_flat)  # [batch*seq, nodes, out_features]

        Wh1 = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)
        Wh2 = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)
        Wh_concat = torch.cat([Wh1, Wh2], dim=-1)

        e = self.leakyrelu(self.a(Wh_concat)).squeeze(-1)
        if adj_mask is not None:
            adj_mask_flat = adj_mask.unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
            e = e.masked_fill(adj_mask_flat == 0, -1e9)

        attention = F.softmax(e, dim=-1)
        attention = self.dropout_layer(attention)

        h_prime = torch.bmm(attention, Wh)
        h_prime = h_prime.reshape(batch_size, seq_len, num_nodes, -1)

        return F.elu(h_prime)

class STGATBlock(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super(STGATBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, self.head_dim, dropout)
            for _ in range(num_heads)
        ])

        self.temporal_conv = nn.Conv2d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=(3, 1),
            padding=(1, 0)
        )
        self.batch_norm = nn.BatchNorm2d(out_features)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x, adj_mask=None):
        residual = x
        if self.residual is not None:
            residual = self.residual(x)

        head_outputs = []
        for att in self.attentions:
            head_outputs.append(att(x, adj_mask))
        x = torch.cat(head_outputs, dim=-1)
        x = x.permute(0, 3, 2, 1)
        x = self.temporal_conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = x.permute(0, 3, 2, 1)
        x = x + residual

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)