import torch
from . import MaskedHeadAttention

batch_size = 1
seq_len = 12
emb_size = 8
head_size = 10
max_seq_len = 24

x = torch.randn(batch_size, seq_len, emb_size)

MHA = MaskedHeadAttention(
    emb_size=emb_size,
    head_size=head_size,
    max_seq_len=max_seq_len
)

res = MHA.forward(x)
print(x.shape, res.shape)
