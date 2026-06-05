import torch.nn as nn
import torch

class PositionalEmbeddings(nn.Module):
    "pos embs"
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.embeddings = nn.Embedding(max_seq_len, emb_size)

    def forward(self, seq_len: int, start_pos: int = 0):
        # need transfer indices to device
        indices = torch.arange(start_pos, seq_len, device=self.embeddings.weight.device)
        return self.embeddings(indices)
