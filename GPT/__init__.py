import json
import os

import torch
from safetensors.torch import save_file, load_file

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
        self.device = device

        self.tokenEmbeddings = TokenEmbeddings(vocab_size=vocab_size, emb_size=emb_size)
        self.positionalEmbeddings = PositionalEmbeddings(max_seq_len=max_seq_len, emb_size=emb_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.decoder_blocks = torch.nn.ModuleList([
            Decoder(
                num_heads=num_heads,
                emb_size=emb_size,
                head_size=head_size,
                max_seq_len=max_seq_len,
                hidden_size=hidden_size,
                dropout=dropout
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
            self,
            x: torch.Tensor,
            max_new_tokens: int,
            do_sample=True,
            temperature: float = 1.0,
            top_p: float = None,
            top_k: int = None
        ):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_x = x[:, -self.max_seq_len:]
                logits = (self(input_x) / temperature)[:, -1, :]
                # logits.shape = (batch_size, vocab_size)

                if do_sample:
                    if top_k is not None:
                        top_k_clamped = min(top_k, logits.size(-1))
                        # treshold findind — the smallest from top_k logits
                        threshold = torch.topk(logits, top_k_clamped, dim=-1).values[..., -1, None]
                        logits = logits.masked_fill(logits < threshold, float('-inf'))

                    if top_p is not None:
                        probs = torch.softmax(logits, dim=-1)
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                        # del tokens when cumsum > top_p
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        # reorganize mask
                        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
                        logits = logits.masked_fill(indices_to_remove, float('-inf'))

                probs = torch.softmax(logits, dim=-1)

                if do_sample:
                    new_token = torch.multinomial(probs, num_samples=1)
                else:
                    new_token = torch.unsqueeze(probs.argmax(dim=-1), 1)

                x = torch.cat((x, new_token), dim=1)
        return x

    def summary(self):
        line = "─" * 56
        print(f"\n{line}")
        print(f"  {'Layer':<30} {'Params':>12}  {'Share':>6}")
        print(line)

        counts = {}
        for name, module in self.named_children():
            if isinstance(module, torch.nn.ModuleList):
                for i, block in enumerate(module):
                    params = sum(p.numel() for p in block.parameters())
                    counts[f"{name}[{i}]"] = params
            else:
                params = sum(p.numel() for p in module.parameters())
                counts[name] = params

        total = sum(counts.values())
        for name, params in counts.items():
            share = params / total * 100
            print(f"  {name:<30} {params:>12,}  {share:>5.1f}%")

        print(line)
        print(f"  {'Total':<30} {total:>12,}  100.0%")
        print(f"  {'Trainable':<30} {sum(p.numel() for p in self.parameters() if p.requires_grad):>12,}")
        print(f"{line}\n")

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        config = {
            "model_type": "GPT",
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "emb_size": self.emb_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_size": self.head_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout.p,
            "temperature": self.temperature,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, save_dir: str, device: str = "cpu"):
        with open(os.path.join(save_dir, "config.json")) as f:
            config = json.load(f)

        model = cls(
            vocab_size=config["vocab_size"],
            max_seq_len=config["max_seq_len"],
            emb_size=config["emb_size"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            head_size=config["head_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            temperature=config["temperature"],
            device=device,
        )

        state_dict = load_file(os.path.join(save_dir, "model.safetensors"), device=device)
        model.load_state_dict(state_dict)
        return model
