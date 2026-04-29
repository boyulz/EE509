"""Baseline training pipeline.

Usage:
    python train.py --config configs/baseline.yaml

The train_model(config) function is the core API consumed by:
- This script (CLI entry point for baseline runs)
- Yuyan's shadow-model training (called repeatedly with different splits)
- Ruoke's DP-SGD wrapper (injects PrivacyEngine into the optimizer)
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import yaml

from data.load_purchase100 import load_purchase100
from splits import load_splits
from model import build_model
from utils.seeding import set_seed


@dataclass
class TrainConfig:
    # Run identification
    experiment_name: str = "baseline"
    seed: int = 42

    # Data: which split file to load and which subset to train on
    split_seed: int = 42
    train_subset: str = "target_train"   # which key in the splits dict
    eval_subset: str = "target_test"     # held-out evaluation subset

    # Override: pass explicit indices instead of named subsets.
    # When set, these take priority over train_subset/eval_subset.
    # Used by shadow.py to train on custom shadow splits.
    train_indices: Optional[np.ndarray] = None
    eval_indices: Optional[np.ndarray] = None

    # Model
    model: dict = field(default_factory=lambda: {
        "input_dim": 600,
        "hidden_dim": 128,
        "num_classes": 100,
        "activation": "tanh",
    })

    # Optimization
    optimizer: str = "sgd"       # "adam" or "sgd"
    lr: float = 0.01
    momentum: float = 0.9        # SGD only
    weight_decay: float = 0.0    # baseline keeps this 0 to encourage overfitting
    batch_size: int = 128
    epochs: int = 100

    # DP config: None for baseline; populated by Ruoke's wrapper
    dp_config: Optional[dict] = None

    # Output
    output_root: str = "runs"


def _git_commit() -> str:
    """Best-effort git commit hash for traceability."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _make_loader(features, labels, indices, batch_size, shuffle):
    """Build a DataLoader from a subset of the full dataset."""
    X = torch.from_numpy(features[indices]).float()
    y = torch.from_numpy(labels[indices]).long()
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate(model, loader, device, criterion):
    """Run model on a loader, return (avg_loss, accuracy)."""
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_count += y.size(0)
    return total_loss / total_count, total_correct / total_count


def _build_optimizer(model, config: TrainConfig):
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def train_model(config: TrainConfig) -> dict:
    """Train one model end-to-end and return a results dict.

    This is the core API. Side effects:
    - Writes per-epoch metrics to runs/{experiment_name}/seed_{seed}/metrics.jsonl
    - Writes final summary to runs/{experiment_name}/seed_{seed}/final.json
    - Saves model weights to runs/{experiment_name}/seed_{seed}/model.pt
    """
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Output directory -----
    run_dir = Path(config.output_root) / config.experiment_name / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")  # truncate any previous run

    # ----- Data -----
    features, labels = load_purchase100()

    # Resolve indices: explicit override takes priority, else use named subset
    if config.train_indices is not None:
        train_idx = config.train_indices
        eval_idx = config.eval_indices
        if eval_idx is None:
            raise ValueError("If train_indices is set, eval_indices must be too.")
    else:
        splits = load_splits(config.split_seed)
        train_idx = splits[config.train_subset]
        eval_idx = splits[config.eval_subset]

    train_loader = _make_loader(
        features, labels, train_idx,
        batch_size=config.batch_size, shuffle=True,
    )
    eval_loader = _make_loader(
        features, labels, eval_idx,
        batch_size=config.batch_size, shuffle=False,
    )

    # ----- Model + optimizer -----
    model = build_model(config.model).to(device)
    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    # ----- DP hook: Ruoke will wrap optimizer + train_loader here later -----
    # if config.dp_config is not None:
    #     model, optimizer, train_loader = attach_privacy_engine(...)

    # ----- Training loop -----
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)
            epoch_correct += (logits.argmax(dim=1) == y).sum().item()
            epoch_count += y.size(0)

        train_loss = epoch_loss / epoch_count
        train_acc = epoch_correct / epoch_count
        eval_loss, eval_acc = _evaluate(model, eval_loader, device, criterion)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(epoch_record) + "\n")

        if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
            print(f"Epoch {epoch:3d} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"eval loss {eval_loss:.4f} acc {eval_acc:.4f}")

    wall_time = time.time() - start_time

    # ----- Save final artifacts -----
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Strip large index arrays from saved config (they're saved separately
    # in split.npz for shadow runs, and aren't JSON-serializable anyway)
    config_for_log = asdict(config)
    config_for_log.pop("train_indices", None)
    config_for_log.pop("eval_indices", None)

    final = {
        "config": config_for_log,
        "git_commit": _git_commit(),
        "device": str(device),
        "wall_time_seconds": wall_time,
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "final_eval_loss": eval_loss,
        "final_eval_acc": eval_acc,
        "generalization_gap": train_acc - eval_acc,
    }
    with open(run_dir / "final.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nDone in {wall_time:.1f}s. Generalization gap: "
          f"{train_acc - eval_acc:.4f} (train {train_acc:.4f} - eval {eval_acc:.4f})")
    print(f"Artifacts saved to: {run_dir}")

    return final


def load_config(path: str) -> TrainConfig:
    """Load a YAML config and merge into TrainConfig defaults."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed in config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed

    train_model(config)