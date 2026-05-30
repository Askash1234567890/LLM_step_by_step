import argparse
import os

from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from GPT1 import GPT1
from train.src.dataset import GetData
from train.src.trainer import Trainer


def load_config(config_dir: str) -> DictConfig:
    return OmegaConf.merge(
        OmegaConf.load(os.path.join(config_dir, "model.yaml")),
        OmegaConf.load(os.path.join(config_dir, "training.yaml")),
        OmegaConf.load(os.path.join(config_dir, "dataset.yaml")),
        OmegaConf.load(os.path.join(config_dir, "logging.yaml")),
        OmegaConf.load(os.path.join(config_dir, "checkpointing.yaml")),
    )


def build_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    train_cfg = config.training
    data_cfg = config.dataset

    pin_memory = train_cfg.device.startswith("cuda")
    num_workers = train_cfg.get("num_workers", 0)

    train_dataset = GetData(data_path=data_cfg.train_path, seq_len=data_cfg.seq_len)
    valid_dataset = GetData(data_path=data_cfg.val_path,   seq_len=data_cfg.seq_len)

    print(f"  train samples: {len(train_dataset):,}")
    print(f"  valid samples: {len(valid_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.train_batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_cfg.valid_batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return train_loader, valid_loader


def main(config_dir: str, resume_from: str = None):
    config = load_config(config_dir)

    model_cfg = config.model
    train_cfg = config.training

    model = GPT1(
        vocab_size=model_cfg.vocab_size,
        max_seq_len=model_cfg.max_seq_len,
        emb_size=model_cfg.emb_size,
        hidden_size=model_cfg.hidden_size,
        num_heads=model_cfg.num_heads,
        head_size=model_cfg.head_size,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        device=train_cfg.device,
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
    parser.add_argument("--config-dir", default="train/configs")
    parser.add_argument("--resume", default=None, help="путь до чекпоинта для resume")
    args = parser.parse_args()

    main(config_dir=args.config_dir, resume_from=args.resume)
