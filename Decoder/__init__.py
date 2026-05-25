from attention.MultiHeadAttention import MultiHeadAttention
from FFN import FeedForward
import torch


class Decoder(torch.nn.Module):
    def __init__(
        self, 
        num_heads: int, 
        emb_size: int, 
        head_size: int, 
        max_seq_len: int, 
        hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.MHA = MultiHeadAttention(
                                num_heads=num_heads,
                                emb_size=emb_size,
                                max_seq_len=max_seq_len,
                                head_size=head_size,
                                dropout=dropout
                            )
        self.ffn = FeedForward(
            emb_size=emb_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        self.norm1 = torch.nn.LayerNorm(emb_size)
        self.norm2 = torch.nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor):
        x_MHA = self.MHA(x)
        x = x + x_MHA

        x = self.norm1(x)

        x_ffn = self.ffn(x)
        x = x + x_ffn

        x = self.norm2(x)
        return x

