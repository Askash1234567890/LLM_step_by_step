import os

import numpy as np
import torch
from torch.utils.data import Dataset


class GetData(Dataset):
    """
    Reads pre-tokenized data from a binary .bin file (int32 numpy array).
    Produced by train/prepare_data.py.

    Each sample is a (input, target) pair of length seq_len,
    where target = input shifted by 1 (next-token prediction).
    """

    def __init__(self, data_path: str, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.data = np.memmap(os.path.expanduser(data_path), dtype=np.int32, mode="r")

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int):
        chunk = torch.from_numpy(
            self.data[idx : idx + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


if __name__ == "__main__":
    dataset = GetData(
        data_path="~/pet_projects/datasets/llm_pretrain/test_pretrain/tokenized_binary_texts/train.bin",
        seq_len=512,
    )
    print(f"samples: {len(dataset):,}")
    x, y = dataset[0]
    print(f"x: {x[:10]}  shape: {x.shape}")
    print(f"y: {y[:10]}  shape: {y.shape}")
