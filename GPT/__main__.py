import torch
from GPT import GPT

vocab_size = 1000
max_seq_len = 100
emb_size = 28
hidden_size = 50
num_heads = 5
head_size = 20
num_layers = 3
dropout = 0.1
device = "mps"


gpt = GPT(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    emb_size=emb_size,
    hidden_size=hidden_size,
    num_heads=num_heads,
    head_size=head_size,
    num_layers=num_layers,
    dropout=dropout,
    device=device
)

batch_size = 1
seq_len = 20

x = torch.randint(0, vocab_size, (batch_size, seq_len)).to("mps")
res = gpt.generate(x, max_new_tokens=10)

print(f"x is {x}")
print(f"res is {res}")