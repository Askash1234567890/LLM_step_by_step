import torch

from embeddings.TokenEmbeddings import TokenEmbeddings
from embeddings.PositionalEmbeddings import PositionalEmbeddings
from Decoder import Decoder


class GPT(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        emb_size: int,
        hidden_size: int,
        num_heads: int,
        head_size: int,
        num_layers: int,
        temperature: float = 1.0,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.temperature = temperature
        self.dropout = dropout
        self.device = device

        self.tokenEmbeddings = TokenEmbeddings(vocab_size=vocab_size, emb_size=emb_size)
        self.positionalEmbeddings = PositionalEmbeddings(max_seq_len=max_seq_len, emb_size=emb_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.decoder_blocks = torch.nn.ModuleList([
            Decoder(
                num_heads = num_heads,
                emb_size = emb_size,
                head_size = head_size,
                max_seq_len = max_seq_len,
                hidden_size = hidden_size,
                dropout = dropout
            ) for _ in range(num_layers)
        ])
        self.linear = torch.nn.Linear(emb_size, vocab_size)
        self.to(device)

    def forward(self, x: torch.Tensor):
        embeddings = self.tokenEmbeddings(x) + self.positionalEmbeddings(x.shape[1])
        output = self.dropout(embeddings)

        for decoder in self.decoder_blocks:
            output = decoder(output)
        
        return self.linear(output)

    def generate(
        self, x: torch.Tensor, 
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0
    ):
        for _ in range(max_new_tokens):
            res = x[:, -self.max_seq_len:]
            logits = self(res) / temperature
            probs = torch.softmax(logits, dim=-1)[:, -1, :]
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.unsqueeze(probs.argmax(dim=-1), 1)
            x = torch.cat((x, next_token), dim=1)
        return x

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)
        
    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model