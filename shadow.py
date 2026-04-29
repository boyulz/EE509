"""
Shadow-model utilities for membership inference attacks.

Provides deterministic, reproducible shadow splits drawn from the shadow_pool
defined in splits.py. Each shadow model gets its own (train, test) pair with
sizes matching the target model (default 10k each), enabling balanced
member/non-member labeling for the attack model.

Shadow splits are sampled with replacement across shadows (i.e., shadow i and
shadow j may share records). This follows Shokri et al. 2017's setup and
allows N to scale beyond what disjoint sampling would permit.
"""

from dataclasses import asdict, replace
from pathlib import Path
import numpy as np

from splits import load_splits
from train import TrainConfig, train_model

# Default shadow split sizes — match the target model exactly
SHADOW_TRAIN_SIZE = 10_000
SHADOW_TEST_SIZE = 10_000

# Master seed used to derive per-shadow seeds. Don't change after experiments
# have started — it would invalidate any saved shadow models.
SHADOW_MASTER_SEED = 20260427


def make_shadow_split(
        shadow_idx: int,
        split_seed: int = 42,
        train_size: int = SHADOW_TRAIN_SIZE,
        test_size: int = SHADOW_TEST_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic (train, test) index pair for one shadow model.

    Args:
        shadow_idx: 0, 1, 2, ... — selects which shadow split to generate.
        split_seed: Which canonical split file to draw shadow_pool from.
        train_size: Number of "member" indices.
        test_size: Number of "non-member" indices.

    Returns:
        (train_indices, test_indices): np.ndarray of int64 indices into the
        full Purchase-100 dataset. Disjoint within a single shadow split.

    Determinism: Same shadow_idx + split_seed always returns the same indices.
    """
    splits = load_splits(split_seed)
    pool = splits["shadow_pool"]

    if train_size + test_size > len(pool):
        raise ValueError(
            f"Requested shadow split ({train_size} + {test_size} = "
            f"{train_size + test_size}) exceeds shadow_pool size ({len(pool)})."
        )

    # Per-shadow RNG: deterministic in shadow_idx, independent across shadows
    rng = np.random.default_rng(SHADOW_MASTER_SEED + shadow_idx)

    # Sample without replacement *within* a single shadow's split
    sampled = rng.choice(pool, size=train_size + test_size, replace=False)
    train_indices = sampled[:train_size].astype(np.int64)
    test_indices = sampled[train_size:].astype(np.int64)

    # Sanity check: train and test must be disjoint within this shadow
    assert len(np.intersect1d(train_indices, test_indices)) == 0

    return train_indices, test_indices


def train_shadow_model(
        shadow_idx: int,
        base_config: TrainConfig | None = None,
        output_root: str = "runs/shadows",
) -> dict:
    """Train one shadow model end-to-end.

    Convenience wrapper around train_model that:
    1. Generates shadow split for the given shadow_idx
    2. Saves the split to disk so the attack code can find it later
    3. Trains using the same architecture/hyperparameters as the target model
    4. Returns the standard train_model results dict

    Args:
        shadow_idx: 0, 1, 2, ...
        base_config: Training config to use. Defaults to baseline.yaml settings
            with experiment_name overridden to "shadow_{idx}".
        output_root: Where to save shadow runs. Default puts them under
            runs/shadows/ to keep them separate from baseline runs.

    Returns:
        Results dict from train_model (final accuracies, gap, etc.).
        Model weights at runs/shadows/shadow_{idx}/seed_{seed}/model.pt.
        Shadow split indices at runs/shadows/shadow_{idx}/split.npz.
    """
    if base_config is None:
        # Import here to avoid circular imports at module load
        from train import load_config
        base_config = load_config("configs/baseline.yaml")

    train_idx, test_idx = make_shadow_split(
        shadow_idx, split_seed=base_config.split_seed
    )

    # Save the shadow split so attack code can load (member, non-member) labels
    run_dir = Path(output_root) / f"shadow_{shadow_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "split.npz", train=train_idx, test=test_idx)

    # Build a config that uses the shadow indices instead of named subsets.
    # NOTE: This requires train.py to support index-based subsets — see below.
    config = replace(
        base_config,
        experiment_name=f"shadows/shadow_{shadow_idx}",
        train_indices=train_idx,
        eval_indices=test_idx,
        output_root="runs",
    )
    return train_model(config)


def load_shadow_split(shadow_idx: int, output_root: str = "runs/shadows") -> dict:
    """Load a previously-generated shadow split from disk.

    Returns dict with keys 'train' and 'test', each an np.ndarray of indices.
    """
    path = Path(output_root) / f"shadow_{shadow_idx}" / "split.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No shadow split at {path}. "
            f"Run train_shadow_model({shadow_idx}) first."
        )
    data = np.load(path)
    return {"train": data["train"], "test": data["test"]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow_idx", type=int, required=True)
    parser.add_argument("--n_shadows", type=int, default=None,
                        help="If set, train shadows from 0 to n_shadows-1")
    args = parser.parse_args()

    if args.n_shadows is not None:
        for i in range(args.n_shadows):
            print(f"\n=== Training shadow {i}/{args.n_shadows - 1} ===")
            train_shadow_model(i)
    else:
        train_shadow_model(args.shadow_idx)