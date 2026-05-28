import math
import os

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: DictConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config

        train_cfg = config.get("training", {})
        self.num_epochs = train_cfg.get("num_epochs", 10)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.grad_accumulation_steps = train_cfg.get("grad_accumulation_steps", 1)
        self.warmup_steps = train_cfg.get("warmup_steps", 100)
        self.device = train_cfg.get("device", "cpu")
        self.device_type = self.device.split(":")[0]  # need for torch.autocast

        # amp и non_blocking только для cuda
        self.use_amp = train_cfg.get("use_amp", False) and self.device.startswith("cuda")
        self.non_blocking = self.device.startswith("cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        log_cfg = config.get("logging", {})
        self.log_every_batch = log_cfg.get("log_every_batch", 100)
        self.use_wandb = log_cfg.get("use_wandb", False)

        ckpt_cfg = config.get("checkpointing", {})
        self.save_dir = ckpt_cfg.get("save_dir", "checkpoints")
        self.save_every_epochs = ckpt_cfg.get("save_every_epochs", 1)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 0.1),
        )

        total_steps = self.num_epochs * len(train_loader) // self.grad_accumulation_steps
        self.scheduler = self._build_scheduler(total_steps)

        if self.use_wandb:
            import wandb
            wandb.init(project=log_cfg.get("project_name", "llm"), config=OmegaConf.to_container(config))

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

        pbar = tqdm(
            self.train_loader,
            desc=f"  train",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
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
            avg_loss = total_loss / (batch_idx + 1)
            lr = self.scheduler.get_last_lr()[0]

            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

            if (batch_idx + 1) % self.log_every_batch == 0 and self.use_wandb:
                import wandb
                wandb.log({"train/loss": avg_loss, "train/lr": lr, "step": global_step})

        return total_loss / len(self.train_loader)

    def _valid_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(
            self.valid_loader,
            desc=f"  valid",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device, non_blocking=self.non_blocking)
                targets = targets.to(self.device, non_blocking=self.non_blocking)

                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model(inputs)
                    loss = self._compute_loss(logits, targets)

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

        return total_loss / len(self.valid_loader)

    def _save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}")
        self.model.save_pretrained(checkpoint_dir)

        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "val_loss": val_loss,
                "config": OmegaConf.to_container(self.config),
            },
            os.path.join(checkpoint_dir, "trainer_state.pt"),
        )

    def load_checkpoint(self, checkpoint_dir: str) -> int:
        from GPT import GPT
        self.model = GPT.from_pretrained(checkpoint_dir, device=self.device)

        trainer_state = torch.load(
            os.path.join(checkpoint_dir, "trainer_state.pt"),
            map_location=self.device,
        )
        self.optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(trainer_state["scheduler_state_dict"])
        self.scaler.load_state_dict(trainer_state["scaler_state_dict"])
        tqdm.write(f"resumed from {checkpoint_dir} (epoch {trainer_state['epoch']})")
        return trainer_state["epoch"]

    def train(self, resume_from: str = None):
        start_epoch = 1

        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1

        epoch_pbar = tqdm(
            range(start_epoch, self.num_epochs + 1),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )

        for epoch in epoch_pbar:
            train_loss = self._train_epoch(epoch)
            val_loss = self._valid_epoch(epoch)

            epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")
            tqdm.write(f"epoch {epoch}/{self.num_epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")

            if self.use_wandb:
                import wandb
                wandb.log({"epoch/train_loss": train_loss, "epoch/val_loss": val_loss, "epoch": epoch})

            if epoch % self.save_every_epochs == 0:
                self._save_checkpoint(epoch, val_loss)
                tqdm.write(f"checkpoint saved → {os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}')}")
