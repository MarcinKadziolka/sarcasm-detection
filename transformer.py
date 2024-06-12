"""
Code from:
https://github.com/pbloem/former/blob/master/former/transformers.py
"""

import torch
from torch import nn
import torch.nn.functional as F

from modules import TransformerBlock
import math


def d(tensor=None):
    if tensor is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if tensor.is_cuda else "cpu"


class Transformer(nn.Module):
    def __init__(
        self,
        emb,
        heads,
        depth,
        seq_length,
        num_tokens,
        max_pool=True,
        dropout=0.0,
        attention_type="default",
    ):
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool
        self.token_embedding = nn.Embedding(
            embedding_dim=emb, num_embeddings=num_tokens
        )

        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    seq_length=seq_length,
                    dropout=dropout,
                    attention_type=attention_type,
                )
            )

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, 1)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        positions = self.pos_embedding(torch.arange(t, device=dev))[None, :, :].expand(
            b, t, e
        )
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = (
            x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)
        )  # pool over the time dimension

        x = self.toprobs(x)

        return F.sigmoid(x).squeeze()
