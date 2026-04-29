"""
Purchase-100 dataset splitting script.

Design principles:
- Deterministic splits generated from a master seed
- Save indices to disk; all team members load the same file
- Validate on load: subsets are disjoint, sizes are correct,
  and the union covers every index in the dataset
"""

from pathlib import Path
import numpy as np
from data.load_purchase100 import load_purchase100

# Split sizes (sum must equal the full dataset size)
SPLIT_SIZES = {
    "target_train":  10_000,
    "target_test":   10_000,
    "val":            5_000,
    "shadow_pool":  172_324,
}

SPLITS_DIR = Path("data/splits")


def make_splits(seed: int) -> dict[str, np.ndarray]:
    """Generate deterministic dataset splits from a seed.

    Returns a dict mapping subset name -> int64 index array.
    """
    features, _ = load_purchase100()
    n_total = len(features)

    expected_total = sum(SPLIT_SIZES.values())
    assert n_total == expected_total, (
        f"Dataset size {n_total} does not match split total {expected_total}. "
        f"Check SPLIT_SIZES."
    )

    # Use an isolated RNG so we don't pollute global random state
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_total)

    splits = {}
    cursor = 0
    for name, size in SPLIT_SIZES.items():
        splits[name] = permutation[cursor:cursor + size].astype(np.int64)
        cursor += size

    _validate_splits(splits, n_total)
    return splits


def _validate_splits(splits: dict[str, np.ndarray], n_total: int) -> None:
    """Sanity checks: disjoint subsets, correct sizes, full coverage."""
    for name, expected_size in SPLIT_SIZES.items():
        assert name in splits, f"Missing subset: {name}"
        assert len(splits[name]) == expected_size, (
            f"{name} has size {len(splits[name])}, expected {expected_size}"
        )

    all_indices = np.concatenate(list(splits.values()))
    assert len(all_indices) == n_total, "Total index count != dataset size"
    assert len(np.unique(all_indices)) == n_total, "Duplicate indices (subsets overlap)"
    assert all_indices.min() == 0 and all_indices.max() == n_total - 1, "Index range invalid"


def save_splits(seed: int, splits: dict[str, np.ndarray]) -> Path:
    """Save splits to disk."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"seed_{seed}.npz"
    np.savez(path, **splits)
    return path


def load_splits(seed: int) -> dict[str, np.ndarray]:
    """Load splits from disk. Errors out if missing (do not silently regenerate)."""
    path = SPLITS_DIR / f"seed_{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. "
            f"Run `python splits.py --seed {seed}` first to generate splits."
        )

    data = np.load(path)
    splits = {name: data[name] for name in SPLIT_SIZES.keys()}

    # Re-validate on load in case the file was modified
    _, _ = load_purchase100()  # ensure dataset is available
    n_total = sum(SPLIT_SIZES.values())
    _validate_splits(splits, n_total)

    return splits


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating splits with seed={args.seed}...")
    splits = make_splits(args.seed)

    print("\nSplit summary:")
    for name, indices in splits.items():
        print(f"  {name:15s}: {len(indices):>7,} records "
              f"(index range [{indices.min():>6}, {indices.max():>6}])")

    path = save_splits(args.seed, splits)
    print(f"\nSaved to: {path}")

    # Verify reload works
    reloaded = load_splits(args.seed)
    print(f"Reload verified, all sanity checks passed.")