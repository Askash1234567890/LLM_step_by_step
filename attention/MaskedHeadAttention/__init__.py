import torch
import math

class MaskedHeadAttention(torch.nn.Module):
    def __init__(
            self, 
            emb_size: int, 
            head_size: int, 
            max_seq_len: int
        ):
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
        ).bool())

    def forward(
            self, 
            x: torch.Tensor, 
            cache: tuple | None = None, 
            use_cache: bool = False
        ):
        "cache = (K_cache, V_cache)"
        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)
        seq_len = K.shape[1]

        if use_cache:
            if cache is not None:
                K_cache, V_cache = cache
            else:
                K_cache = torch.zeros(x.shape[0], 0, self.head_size, device=x.device)
                V_cache = torch.zeros(x.shape[0], 0, self.head_size, device=x.device)
            K = torch.cat([K_cache, K], dim=1)
            V = torch.cat([V_cache, V], dim=1)

            if K.shape[1] > self.max_seq_len:
                raise ValueError(
                    f"KV-cache overflow: {K.shape[1]} tokens exceed max_seq_len={self.max_seq_len}"
                )

        attention = Q @ K.transpose(-1, -2) / math.sqrt(self.head_size)

        if use_cache:
            T_q    = Q.shape[1]
            T_k    = K.shape[1]        # T_past + T_q
            T_past = T_k - T_q

            mask = torch.ones(T_q, T_k, dtype=torch.bool, device=x.device)
            mask[:, T_past:] = self.Mask[:T_q, :T_q]
            masked_attention = attention.masked_fill(~mask, float('-inf'))
        else:
            masked_attention = attention.masked_fill(
                ~self.Mask[:seq_len, :seq_len], 
                float('-inf')
            )

        res_attention = torch.softmax(masked_attention, dim=-1)
        returned_cache = None
        if use_cache:
            returned_cache = (K, V)
        
        return (res_attention @ V, returned_cache)
