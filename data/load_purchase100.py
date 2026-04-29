import numpy as np
import pandas as pd
from pathlib import Path
import hashlib

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data/raw/dataset_purchase"
CACHE_PATH = REPO_ROOT / "data/processed/purchase100.npz"
EXPECTED_SHAPE = (197324, 600)
EXPECTED_NUM_CLASSES = 100


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_dataset(features: np.ndarray, labels: np.ndarray) -> None:
    if features.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    if labels.min() != 0 or labels.max() != EXPECTED_NUM_CLASSES - 1:
        raise ValueError(
            f"Unexpected label range: [{labels.min()}, {labels.max()}]"
        )
    if not set(np.unique(features)).issubset({0.0, 1.0}):
        raise ValueError("Features should be binary")


def load_purchase100(use_cache: bool = True):
    """Load Purchase-100. Returns (features, labels):
       features: float32, shape (N, 600), values in {0, 1}
       labels:   int64,   shape (N,),     values in {0, ..., 99}
    """
    if use_cache and CACHE_PATH.exists():
        with np.load(CACHE_PATH) as data:
            features = data["features"]
            labels = data["labels"]
        _validate_dataset(features, labels)
        return features, labels

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            f"Download and extract dataset_purchase.tgz first."
        )

    # First column is the label (1-100); the next 600 columns are features
    df = pd.read_csv(DATA_PATH, header=None)
    labels = df.iloc[:, 0].values.astype(np.int64) - 1  # convert to 0-99
    features = df.iloc[:, 1:].values.astype(np.float32)

    _validate_dataset(features, labels)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, features=features, labels=labels)
    return features, labels


if __name__ == "__main__":
    X, y = load_purchase100()
    print(f"Features: shape={X.shape}, dtype={X.dtype}, min={X.min()}, max={X.max()}")
    print(f"Labels:   shape={y.shape}, dtype={y.dtype}, min={y.min()}, max={y.max()}")
    print(f"Samples per class (first 5): {np.bincount(y)[:5]}")
    print(f"Raw file MD5: {_md5(DATA_PATH)}")
