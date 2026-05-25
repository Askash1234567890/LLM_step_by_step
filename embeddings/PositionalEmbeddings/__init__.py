import torch.nn as nn
import torch

class PositionalEmbeddings(nn.Module):
    "pos embs"
    def __init__(self, max_seq_len, emb_size):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.embeddings = nn.Embedding(max_seq_len, emb_size)

    def forward(self, seq_len: int):
        indices = torch.arange(seq_len)
        return self.embeddings(indices)
