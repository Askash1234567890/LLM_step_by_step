import torch
from ..MaskedHeadAttention import MaskedHeadAttention


class MultiHeadAttention(torch.nn.Module):
    "multihead attn"
    def __init__(
            self, 
            num_heads: int, 
            emb_size: int, 
            head_size: int, 
            max_seq_len: int, 
            dropout: float = 0.1
        ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.Heads = torch.nn.ModuleList(
            [
                MaskedHeadAttention(
                    emb_size=emb_size,
                    head_size=head_size,
                    max_seq_len=max_seq_len
                )
                for _ in range(num_heads)
            ]
        )
        self.Mo = torch.nn.Linear(num_heads * head_size, emb_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
            self, 
            x: torch.Tensor, 
            cache: list[tuple] | None = None, 
            use_cache: bool = False
        ):

        n = len(self.Heads)
        if cache is None:
            cache = [None] * n

        outputs, new_cache = [None] * n, [None] * n
        for i, MHA in enumerate(self.Heads):
            out, head_cache = MHA(x, cache=cache[i], use_cache=use_cache)
            outputs[i] = out
            new_cache[i] = head_cache

        o_concat = torch.cat(outputs, dim=-1)
        O = self.dropout(self.Mo(o_concat))

        return O, new_cache
