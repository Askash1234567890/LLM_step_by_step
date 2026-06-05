import torch
from . import MultiHeadAttention

batch_size = 1
seq_len = 12
emb_size = 8
head_size = 10
max_seq_len = 24
num_heads = 5
dropout = 0.1

x = torch.randn(batch_size, seq_len, emb_size)

MHA = MultiHeadAttention(
    num_heads=num_heads,
    emb_size=emb_size,
    head_size=head_size,
    max_seq_len=max_seq_len,
    dropout=dropout
)

cache, use_cache = None, True
res, res_cache = MHA(
    x=x,
    cache=cache,
    use_cache=use_cache
)
print(x.shape, res.shape)
