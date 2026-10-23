import math

import torch
from torch import nn


class EmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, d_embed, d_model, max_seq_length=5000, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0

        self.d_model = d_model
        self.scaling = math.sqrt(self.d_model)

        self.d_embed = d_embed
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.projection = nn.Linear(self.d_embed, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding = self.create_positional_embedding(
            max_seq_length, d_model
        )
        self.register_buffer(
            "positional_encoding", self.positional_encoding, persistent=False
        )

    @staticmethod
    def create_positional_embedding(seq_length, d_model, device=None):
        position = torch.arange(seq_length, device=device).unsqueeze(1).float()
        division_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        empty_tensor = torch.zeros(seq_length, d_model, device=device)

        empty_tensor[:, 0::2] = torch.sin(position * division_term)
        empty_tensor[:, 1::2] = torch.cos(position * division_term)

        return empty_tensor.unsqueeze(0)

    def forward(self, x):
        assert x.dtype == torch.long, (
            f"Input tensor must have dtype torch.long, got {x.dtype}"
        )

        seq_length = x.size(1)

        if seq_length > self.positional_encoding.size(1):
            raise ValueError(
                f"Sequência de tamanho {seq_length} excede max_seq_length "
                f"({self.positional_encoding.size(1)})"
            )

        token_embedding = self.projection(self.embedding(x)) * self.scaling

        positional_encoding = self.positional_encoding[:, :seq_length]
        output = self.layernorm(token_embedding + positional_encoding)

        return self.dropout(output)
