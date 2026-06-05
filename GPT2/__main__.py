import torch
from GPT1 import GPT1

vocab_size = 1000
max_seq_len = 100
emb_size = 28
hidden_size = 50
num_heads = 5
head_size = 20
num_layers = 3
temperature = 1.0
dropout = 0.1
do_sample = True
device = "mps"

# determe the model init
torch.manual_seed(42)

gpt = GPT1(
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    emb_size=emb_size,
    hidden_size=hidden_size,
    num_heads=num_heads,
    head_size=head_size,
    num_layers=num_layers,
    temperature=temperature,
    dropout=dropout,
    device=device
)

batch_size = 1
seq_len = 20

x = torch.tensor([[ 54, 628, 429, 576, 634, 488, 963, 948, 137, 691, 606, 641, 789, 249, 777, 282, 217, 137, 871, 495]]).to(device)
res = gpt.generate(
    x=x, 
    max_new_tokens=10,
    temperature=0.5,
    do_sample=True
)

print(f"x is {x}")
print(f"res is {res}")