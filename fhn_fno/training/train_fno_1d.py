import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional

# amp import location changed between torch versions
try:
    from torch.cuda.amp import GradScaler, autocast

    LEGACY_AMP = True
except ImportError:
    try:
        from torch.amp import GradScaler, autocast

        LEGACY_AMP = False
    except ImportError:

        class GradScaler:
            def __init__(self, *args, **kwargs):
                pass

            def scale(self, loss):
                return loss

            def step(self, optimizer):
                optimizer.step()

            def update(self):
                pass

            def unscale_(self, optimizer):
                pass

        class autocast:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        LEGACY_AMP = None

sys.path.append(".")

from fhn_fno.config import Config, TrainingConfig
from fhn_fno.data.dataset import FHNOperatorDataset
from fhn_fno.models.fno import FNO
from fhn_fno.eval.metrics import relative_l2_error, compute_memory_usage


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    config: TrainingConfig,
    scaler: Optional["GradScaler"] = None,
    epoch: int = 0,
):
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        x = batch["input"].to(device)
        y = batch["target"].to(device)

        optimizer.zero_grad()

        if config.use_amp and device != "cpu" and LEGACY_AMP is not None:
            if LEGACY_AMP:
                with autocast():
                    pred = model(x)
                    loss = criterion(pred, y)
            else:
                with autocast("cuda"):
                    pred = model(x)
                    loss = criterion(pred, y)

            if scaler is not None and LEGACY_AMP is not None:
                scaler.scale(loss).backward()

                if config.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip
                    )
                optimizer.step()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / n_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    config: TrainingConfig,
):
    model.eval()
    total_loss = 0.0
    total_rel_error_u = 0.0
    total_rel_error_v = 0.0
    n_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            x = batch["input"].to(device)
            y = batch["target"].to(device)

            if config.use_amp and device != "cpu" and LEGACY_AMP is not None:
                if LEGACY_AMP:
                    with autocast():
                        pred = model(x)
                        loss = criterion(pred, y)
                else:
                    with autocast("cuda"):
                        pred = model(x)
                        loss = criterion(pred, y)
            else:
                pred = model(x)
                loss = criterion(pred, y)

            total_loss += loss.item()

            rel_err_u = relative_l2_error(pred[:, 0:1], y[:, 0:1])
            rel_err_v = relative_l2_error(pred[:, 1:2], y[:, 1:2])
            total_rel_error_u += rel_err_u.item()
            total_rel_error_v += rel_err_v.item()

    avg_loss = total_loss / n_batches
    avg_rel_error_u = total_rel_error_u / n_batches
    avg_rel_error_v = total_rel_error_v / n_batches

    return avg_loss, avg_rel_error_u, avg_rel_error_v


def train(
    config: Config,
    data_file: str,
    device: str = None,
    checkpoint_dir: str = None,
    epochs: int = None,
    target_loss: float = None,
):

    if device:
        config.training.device = device
    if checkpoint_dir:
        config.training.checkpoint_dir = checkpoint_dir
    if epochs:
        config.training.epochs = epochs

    device = config.training.device

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_dataset = FHNOperatorDataset(
        data_file, mode="single_step", train=True, device="cpu"
    )
    val_dataset = FHNOperatorDataset(
        data_file, mode="single_step", train=False, device="cpu"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers if device != "cpu" else 0,
        pin_memory=config.training.pin_memory and device != "cpu",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers if device != "cpu" else 0,
        pin_memory=config.training.pin_memory and device != "cpu",
    )

    model = FNO(
        modes=config.model.modes,
        width=config.model.width,
        n_layers=config.model.n_layers,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        dim=train_dataset.dim,
        use_positional=config.model.use_positional,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.training.loss_type == "l2":
        criterion = nn.MSELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    if config.training.use_scheduler:
        if config.training.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.training.epochs
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    else:
        scheduler = None

    scaler = None
    if config.training.use_amp and device != "cpu" and LEGACY_AMP is not None:
        if LEGACY_AMP:
            scaler = GradScaler()
        else:
            scaler = GradScaler("cuda")

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.training.epochs):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.training,
            scaler,
            epoch,
        )

        if epoch % config.training.val_every == 0:
            val_loss, val_rel_u, val_rel_v = validate(
                model, val_loader, criterion, device, config.training
            )

            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Rel L2 u: {val_rel_u:.4f}, Rel L2 v: {val_rel_v:.4f}"
            )

            if target_loss is not None and val_loss <= target_loss:
                print(
                    f"\n🎯 Target loss of {target_loss:.4f} reached! "
                    f"Current val loss: {val_loss:.4f}"
                )
                print(f"Stopping training at epoch {epoch}")

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": config.to_dict(),
                    },
                    checkpoint_dir / "target_reached_model.pt",
                )
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "config": config.to_dict(),
                    },
                    checkpoint_dir / "best_model.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= config.training.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if epoch % config.training.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "config": config.to_dict(),
                },
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    print("Done training")
    return model


def main():
    DATA_FILE = "data/fhn_1d_128.h5"
    EPOCHS = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "checkpoints/"
    TARGET_LOSS = 0.01

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {DEVICE}")

    config = Config()

    if BATCH_SIZE:
        config.training.batch_size = BATCH_SIZE
    if LEARNING_RATE:
        config.training.lr = LEARNING_RATE

    # FNO uses complex ops, which break under half precision
    config.training.use_amp = False

    if LEGACY_AMP is None:
        print("Warning: AMP not available, using regular precision")
    elif config.training.use_amp:
        print("Mixed precision training enabled")
    else:
        print("Regular precision training")

    train(config, DATA_FILE, DEVICE, CHECKPOINT_DIR, EPOCHS, TARGET_LOSS)


if __name__ == "__main__":
    main()
