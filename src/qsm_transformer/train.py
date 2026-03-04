from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from qsm_transformer.data.synthetic_qsm import SyntheticQSMConfig, SyntheticQSMDataset
from qsm_transformer.losses.qsm_losses import QSMCompositeLoss
from qsm_transformer.models.factory import build_model
from qsm_transformer.utils.metrics import mae, psnr


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train(optimizer is not None)
    total_loss = 0.0
    total_mae = 0.0
    total_psnr = 0.0

    for batch in tqdm(loader, leave=False):
        field = batch["field"].to(device)
        target = batch["susceptibility"].to(device)

        pred = model(field)
        loss = criterion(pred, target)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_mae += mae(pred.detach(), target).item()
        total_psnr += psnr(pred.detach(), target).item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "mae": total_mae / n,
        "psnr": total_psnr / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QSM reconstruction models")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = SyntheticQSMConfig(**cfg["dataset"])
    train_ds = SyntheticQSMDataset(length=cfg["train"]["train_samples"], config=dataset_cfg)
    val_ds = SyntheticQSMDataset(length=cfg["train"]["val_samples"], config=dataset_cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = build_model(cfg["model"]["name"], **cfg["model"].get("args", {})).to(device)
    criterion = QSMCompositeLoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_psnr={val_metrics['psnr']:.2f}, val_mae={val_metrics['mae']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            ckpt = out_dir / f"best_{cfg['model']['name']}.pt"
            torch.save(model.state_dict(), ckpt)


if __name__ == "__main__":
    main()
