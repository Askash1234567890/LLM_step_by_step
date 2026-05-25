import torch
from ..MaskedHeadAttention import MaskedHeadAttention

class MultiHeadAttention():
    "multihead attn"
    def __init__(self, num_heads, emb_size, head_size, max_seq_len, dropout):
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

    def forward(self, x: torch.Tensor):
        outputs = [MHA(x) for MHA in self.Heads]
        o_concat = torch.cat(outputs, dim=-1)
        O = self.Mo(o_concat)

        return self.dropout(O)
