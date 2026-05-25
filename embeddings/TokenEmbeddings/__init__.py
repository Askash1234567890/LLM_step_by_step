import torch.nn as nn
import torch

class TokenEmbeddings(nn.Module):
    "token embeddings"
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor):
        return self.embeddings(x)
