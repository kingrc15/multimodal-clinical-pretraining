import math
import torch
from torch import nn


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_features,
        embed_dim,
        num_heads,
        dropout=0,
        num_layers=6,
        use_pos_emb=False,
        activation=nn.GELU(),
        mask_rate=0.0,
    ):
        super().__init__()

        self.use_pos_emb = use_pos_emb
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.mask_rate = mask_rate

        self.cls_token = nn.Parameter(torch.randn(1, 1, 76))

        self.embedding = nn.Linear(n_features, embed_dim)
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim,
                num_heads,
                batch_first=True,
                dropout=dropout,
                activation=activation,
            ),
            num_layers,
        )

        self.pos_embedding = PositionalEncoding(embed_dim, dropout=0)

    def forward(self, x):
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        if self.training:
            cls_token = cls_token.masked_fill(
                torch.rand_like(cls_token[:, 0, 0, None, None].float())
                < self.mask_rate,
                0,
            )
        x = torch.cat((self.cls_token.repeat(x.size(0), 1, 1), x), dim=1)
        mask = (x == float("inf"))[:, :, 0]
        x.masked_fill_(mask[:, :, None], 0.0)

        z = self.embedding(x)
        z = self.pos_embedding(z, mask[:, :, None])
        z = self.trans(z, src_key_padding_mask=mask)

        return z


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x, mask=None):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]

        if mask is not None:
            x.masked_fill_(mask, 0.0)

        return self.dropout(x)
