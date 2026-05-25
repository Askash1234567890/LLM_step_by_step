import torch

class FeedForward(torch.nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, hidden_size = int | None):
        super().__init__()
        if hidden_size is None:
            hidden_size = 4 * emb_size

        self.fc1 = torch.nn.Linear(emb_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, emb_size)

        self.act1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        
        return self.dropout(x)