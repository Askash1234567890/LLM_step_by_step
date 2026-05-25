import torch
from . import FeedForward

batch_size = 3
seq_len = 12
emb_size = 8
dropout = 0.1

x = torch.randn(batch_size, seq_len, emb_size)

net = FeedForward(
    emb_size=emb_size,
    dropout=dropout,
    hidden_size=8 * emb_size
)
res = net.forward(x)

print(res, res.shape)