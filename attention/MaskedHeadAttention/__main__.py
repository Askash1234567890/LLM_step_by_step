import torch
from . import MaskedHeadAttention

batch_size = 2
seq_len    = 6
emb_size   = 8
head_size  = 10
max_seq_len = 24

torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, emb_size)

mha = MaskedHeadAttention(emb_size=emb_size, head_size=head_size, max_seq_len=max_seq_len)
mha.eval()

# --- 1. no KV-cache ---
with torch.no_grad():
    out_full, _ = mha(x, use_cache=False)          # (B, T, head_size)

# --- 2. with KV-cache ---
outputs = []
cache = None
with torch.no_grad():
    for t in range(seq_len):
        token = x[:, t : t + 1, :]                     # (B, 1, emb_size)
        out, cache = mha(token, cache=cache, use_cache=True)
        outputs.append(out)

out_cached = torch.cat(outputs, dim=1)             # (B, T, head_size)

# --- 3. compare ---
match = torch.allclose(out_full, out_cached, atol=1e-6)
max_diff = (out_full - out_cached).abs().max().item()

print(f"out_full   shape : {out_full.shape}")
print(f"out_cached shape : {out_cached.shape}")
print(f"outputs match    : {match}  (max diff: {max_diff:.2e})")
print(f"cache size after : {cache[0].shape[1]} tokens")

assert match, "KV-cache output differs from full forward — something is wrong!"
