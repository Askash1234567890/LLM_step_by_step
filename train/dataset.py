import torch
from torch.utils.data import Dataset

class GetData(Dataset):
    def __init__(
        self, 
        data: list[int],
        seq_len: int,
        device: str = "cpu"
    ):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx: int):
        x = torch.tensor(self.data[idx: idx + self.seq_len], dtype=torch.long, device=self.device)
        y = torch.tensor(self.data[idx + 1: idx + self.seq_len + 1], dtype=torch.long, device=self.device)

        return x, y

if __name__ == "__main__":
    getData = GetData(data=[1, 2, 3, 4], seq_len=2, device="mps")
    print(getData[0])