import torch
import math

class MaskedHeadAttention(torch.nn.Module):
    def __init__(self, emb_size, head_size, max_seq_len):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        # x.size = (batch_size, seq_len, emb_size)
        # K must has shape (seq_len, head_size); K = x * Wk
        self.Wk = torch.nn.Linear(emb_size, head_size)
        self.Wq = torch.nn.Linear(emb_size, head_size)
        self.Wv = torch.nn.Linear(emb_size, head_size)
        # need for switch data to device
        self.register_buffer('Mask', torch.tril(
            torch.ones((max_seq_len, max_seq_len))
        ))

    def forward(self, x: torch.Tensor):
        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)
        seq_len = K.shape[1]

        attention = K @ Q.transpose(-1, -2) / math.sqrt(self.head_size)
        masked_attention = torch.mul(attention, self.Mask[:seq_len, :seq_len])
        masked_attention[masked_attention == 0] = float("-inf")

        masked_attention = torch.softmax(masked_attention, dim=-1)
        return masked_attention @ V
