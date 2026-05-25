import torch
from . import Decoder


num_heads = 5
emb_size = 28 
head_size = 15 
max_seq_len = 10
hidden_size = 3 * emb_size
dropout: float = 0.1

decoder = Decoder(
    num_heads = num_heads,
    emb_size = emb_size,
    head_size = head_size,
    max_seq_len = max_seq_len,
    hidden_size = hidden_size,
    dropout = dropout
).to("mps")

batch_size = 2
seq_len = 7

x = torch.randn(batch_size, seq_len, emb_size).to("mps")

print(decoder.forward(x))