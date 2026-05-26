import math
import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config

        train_cfg = config["training"]
        self.num_epochs = train_cfg.get("num_epochs", 10)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.grad_accumulation_steps = train_cfg.get("grad_accumulation_steps", 1)
        self.warmup_steps = train_cfg.get("warmup_steps", 100)
        self.device = train_cfg.get("device", "cpu")
        self.device_type = self.device.split(":")[0]  # need for torch.autocast

        # amp и pin_memory только для cuda
        self.use_amp = train_cfg.get("use_amp", False) and self.device.startswith("cuda")
        self.non_blocking = self.device.startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        log_cfg = config["logging"]
        self.log_every_batch = log_cfg.get("log_every_batch", 100)
        self.use_wandb = log_cfg.get("use_wandb", False)

        ckpt_cfg = config["checkpointing"]
        self.save_dir = ckpt_cfg.get("save_dir", ".")
        self.save_every_epochs = ckpt_cfg.get("save_every_epochs", 5)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        total_steps = self.num_epochs * len(train_loader) // self.grad_accumulation_steps
        self.scheduler = self._build_scheduler(total_steps)

        if self.use_wandb:
            import wandb
            wandb.init(project=log_cfg["project_name"], config=config)

        os.makedirs(self.save_dir, exist_ok=True)

    def _build_scheduler(self, total_steps: int) -> LambdaLR:
        warmup = self.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, S, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * S, V),
            targets.reshape(B * S),
        )

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        global_step = 0

        self.optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device, non_blocking=self.non_blocking)
            targets = targets.to(self.device, non_blocking=self.non_blocking)

            # TODO: there are some specific cases with amp_dtype in CPU and GPU - check it
            with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                logits = self.model(inputs)
                loss = self._compute_loss(logits, targets) / self.grad_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * self.grad_accumulation_steps

            if (batch_idx + 1) % self.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                print(f"epoch {epoch} | step {batch_idx + 1} | loss {avg_loss:.4f} | lr {lr:.2e}")

                if self.use_wandb:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/lr": lr, "step": global_step})

        return total_loss / len(self.train_loader)

    def _valid_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.valid_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model(inputs)
                    loss = self._compute_loss(logits, targets)

                total_loss += loss.item()

        return total_loss / len(self.valid_loader)

    def _save_checkpoint(self, epoch: int, val_loss: float):
        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )
        print(f"checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"resumed from {path} (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]

    def train(self, resume_from: str = None):
        start_epoch = 1

        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1

        for epoch in range(start_epoch, self.num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._valid_epoch(epoch)

            print(f"\nepoch {epoch}/{self.num_epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}\n")

            if self.use_wandb:
                import wandb
                wandb.log({"epoch/train_loss": train_loss, "epoch/val_loss": val_loss, "epoch": epoch})

            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, val_loss)
