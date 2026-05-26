import argparse

import yaml
import torch
from torch.utils.data import DataLoader, random_split

from GPT import GPT
from train.dataset import GetData
from train.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    train_cfg = config["training"]
    data_cfg = config["dataset"]

    pin_memory = train_cfg["device"].startswith("cuda")

    # TODO: change to real data
    data: list[int] = []

    dataset = GetData(data=data, seq_len=data_cfg["seq_len"])

    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["train_batch_size"],
        shuffle=True,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=data_cfg["valid_batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader


def main(config_path: str, resume_from: str = None):
    config = load_config(config_path)

    model_cfg = config["model"]
    train_cfg = config["training"]

    model = GPT(
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        emb_size=model_cfg["emb_size"],
        hidden_size=model_cfg["hidden_size"],
        num_heads=model_cfg["num_heads"],
        head_size=model_cfg["head_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        device=train_cfg["device"],
    )

    train_loader, valid_loader = build_dataloaders(config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config,
    )

    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train/configs/base.yaml")
    parser.add_argument("--resume", default=None, help="путь до чекпоинта для resume")
    args = parser.parse_args()

    main(config_path=args.config, resume_from=args.resume)
