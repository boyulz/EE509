# EEP 509 Project: Membership Inference Attacks and DP-SGD Defenses

Spring 2026 course project for EEP 595 (Privacy-Preserving Machine Learning).
We study the privacy–utility tradeoff of DP-SGD as a defense against
membership inference attacks (MIAs) on a Purchase-100 classifier.

## Team

| Member | Role | Owns |
|---|---|---|
| Leslie Zhang | Baseline & infrastructure | Data pipeline, baseline MLP, training harness, seeding/logging |
| Yuyan Ding | Attacks | Loss/threshold MIA, shadow-model MIA, attack evaluation |
| Ruoke Zhang | Defense & analysis | DP-SGD integration, privacy accounting, ε sweep, plots |

## Project overview

**Threat model.** A data holder trains a classifier on sensitive records and
exposes it as a black-box prediction API. An external adversary with query
access tries to determine whether a target record was in the training set
(membership inference). Overfitting creates exploitable gaps between member
and non-member behavior.

**Research question.** How much does DP-SGD reduce MIA risk in this setting,
and what utility cost does it impose? We measure utility (test accuracy) and
privacy leakage (MIA AUC, advantage, and TPR @ FPR=1%) across
ε ∈ {0.5, 1, 4, 8, ∞}, δ = 1/|train|.

**Approach.** Train a baseline MLP on Purchase-100 (deliberately overfit, no
regularization). Run two MIAs (loss/threshold and shadow-model) against it.
Retrain with Opacus DP-SGD across the ε sweep. Re-run both attacks against
every DP model. Aggregate over ≥3 seeds.

## Setup

Python 3.8+ required.

```bash

# 1. Download Purchase-100 (Shokri lab preprocessed version)
mkdir -p data/raw
cd data/raw
wget https://github.com/privacytrustlab/datasets/raw/master/dataset_purchase.tgz
tar -xzf dataset_purchase.tgz
cd ../..

# 2. Verify the loader works (also builds the .npz cache)
python data/load_purchase100.py
```

The loader prints the raw file's MD5. It should match:

```
dd892f2cedf8b07025d016df49a2eb4f
```

If it doesn't match, the download is corrupted — re-fetch.

After loading you should see:

- `features`: shape `(197324, 600)`, dtype `float32`, values in `{0, 1}`
- `labels`: shape `(197324,)`, dtype `int64`, values in `{0, ..., 99}`
  (note: original labels are 1–100; the loader converts to 0-indexed)

## Dataset splits

We use a **fixed split shared by all team members**. Don't regenerate splits
independently — load the committed file. Index counts:

| Subset | Size | Purpose |
|---|---|---|
| `target_train` | 10,000 | Victim model training set (MIA members) |
| `target_test` | 10,000 | Held-out evaluation set (MIA non-members) |
| `val` | 5,000 | Hyperparameter tuning only |
| `shadow_pool` | 172,324 | Pool from which Yuyan carves shadow datasets |

`target_train` and `target_test` are equal-sized so the MIA evaluation set is
balanced (random guessing → 50% accuracy, AUC well-defined).

**On seeds.** We hold splits fixed across training-seed variation. The "≥3
seeds" requirement applies to model training only (weight init, batch order),
not to which records are members. Error bars therefore reflect training
randomness with the membership assignment held constant.

To regenerate the canonical split (only needed if `data/splits/seed_42.npz`
is somehow lost):

```bash
python splits.py --seed 42
```

## Running the baseline

```bash
python train.py --config configs/baseline.yaml
```

Outputs land in `runs/baseline/seed_42/`:

- `metrics.jsonl` — per-epoch train/eval loss and accuracy
- `final.json` — final metrics, config, git commit, wall time
- `model.pt` — model weights (use for downstream attacks)

Common training workflows:

```bash
# Single run
python train.py --config configs/baseline.yaml

# Multi-seed run
python train.py --config configs/baseline.yaml --seeds 1 2 3 42

# Explicitly overwrite an existing run
python train.py --config configs/baseline.yaml --seed 42 --overwrite

# Rebuild summary files from existing per-seed logs
python train.py --config configs/baseline.yaml --aggregate-only
```

To run additional seeds one by one:

```bash
python train.py --config configs/baseline.yaml --seed 1
python train.py --config configs/baseline.yaml --seed 2
python train.py --config configs/baseline.yaml --seed 3
```

For multi-seed experiments, aggregate outputs are written to:

- `runs/baseline/summary.json` — machine-readable aggregate statistics
- `runs/baseline/summary.csv` — per-seed results table for downstream analysis

**Expected behavior.** The baseline is intentionally unregularized so it
overfits — overfitting is what makes MIA work and what DP-SGD must defend.
Healthy targets:

- Train accuracy: 95%+
- Eval accuracy: 50–65%
- Generalization gap (train − eval): at least 0.20

If the gap is too small, the baseline isn't a useful MIA target. Increase
epochs or check for unintended regularization before proceeding.

## Programmatic interfaces (for Yuyan and Ruoke)

The training pipeline is built around a single function: `train_model(config)`
in `train.py`. Both attacks and defenses plug in around it.

### Loading the dataset

```python
from data.load_purchase100 import load_purchase100

features, labels = load_purchase100()
# features: np.ndarray, shape (197324, 600), float32
# labels:   np.ndarray, shape (197324,), int64
```

The first call parses the CSV (~30s) and caches to
`data/processed/purchase100.npz`. Subsequent calls are instant.

### Loading splits

```python
from splits import load_splits

splits = load_splits(seed=42)
# splits is a dict with keys: target_train, target_test, val, shadow_pool
# each value is an np.ndarray of int64 indices into the full dataset

X_train = features[splits["target_train"]]
y_train = labels[splits["target_train"]]
```

`load_splits` validates on read (disjointness, sizes, full coverage), so you
get an immediate error if the file is corrupted.

### Loading a trained model

```python
import torch
from model import build_model

config = {"input_dim": 600, "hidden_dim": 128, "num_classes": 100, "activation": "tanh"}
model = build_model(config)
model.load_state_dict(torch.load("runs/baseline/seed_42/model.pt"))
model.eval()

# Forward pass returns raw logits (no softmax applied)
with torch.no_grad():
    logits = model(torch.from_numpy(X_train).float())
    probs = torch.softmax(logits, dim=1)
```

### Training a model programmatically

```python
from train import TrainConfig, train_model

config = TrainConfig(
    experiment_name="my_experiment",
    seed=42,
    train_subset="target_train",
    eval_subset="target_test",
    epochs=50,
)
results = train_model(config)
# results contains final accuracies, generalization gap, wall time, etc.
```

### Notes for Yuyan (attacks)

- **Shadow-model training.** Use `train_model(config)` repeatedly with
  different splits of `shadow_pool`. We can add a `train_subset_indices`
  override if you need finer control than the named-subset interface — let me
  know what API you'd prefer.
- **Model outputs.** Models return raw logits. Apply softmax yourself if your
  attack needs probabilities (loss-based attacks usually want logits anyway
  for numerical stability).
- **Members vs. non-members.** For evaluating attacks against the baseline:
  members = `splits["target_train"]`, non-members = `splits["target_test"]`.
  Both are size 20,000 → balanced evaluation.
- **Reproducibility.** Always call `set_seed(seed)` at the start of any
  attack script. Import: `from utils.seeding import set_seed`.


### Shadow models (for Yuyan)

Train shadow models for the Shokri et al. 2017 attack:

```bash
python shadow.py --n_shadows 10   # train shadow_0 through shadow_9
```

For each shadow model, you'll find:
- `runs/shadows/shadow_{i}/split.npz` — indices used (train, test)
- `runs/shadows/shadow_{i}/seed_{seed}/model.pt` — trained weights

To use in attack code:

```python
from shadow import load_shadow_split
from data.load_purchase100 import load_purchase100
from model import build_model
import torch

features, labels = load_purchase100()

for i in range(10):
    # Load this shadow's data
    split = load_shadow_split(i)
    member_idx, nonmember_idx = split["train"], split["test"]
    
    # Load this shadow's trained model
    model = build_model({"input_dim": 600, "hidden_dim": 128, 
                          "num_classes": 100, "activation": "tanh"})
    model.load_state_dict(torch.load(f"runs/shadows/shadow_{i}/seed_42/model.pt"))
    model.eval()
    
    # Get logits for members and non-members → use as attack training data
    with torch.no_grad():
        member_logits = model(torch.from_numpy(features[member_idx]).float())
        nonmember_logits = model(torch.from_numpy(features[nonmember_idx]).float())
    # ... build attack training set with labels (1=member, 0=non-member)
```

Shadow splits are deterministic in `shadow_idx` — re-running gives identical
indices. The number of shadows is your choice; Shokri 2017 used 20 for
Purchase, but 10 is usually enough.


### Notes for Ruoke (defense)

- **DP-SGD integration.** `train.py` has a hook marked
  `# DP hook: Ruoke will wrap optimizer + train_loader here later`. Wrap
  `model`, `optimizer`, and `train_loader` with Opacus' `PrivacyEngine` there.
- **Config plumbing.** The `dp_config` field in `TrainConfig` is reserved for
  you. Suggested keys: `target_epsilon`, `target_delta`, `max_grad_norm`,
  `noise_multiplier` (or compute it from ε, δ via Opacus' accountant).
- **Pinning versions.** Please decide on PyTorch + Opacus versions early and
  add them to `requirements.txt`. Opacus is finicky about PyTorch versions.
- **Architecture compatibility.** The baseline MLP has no BatchNorm, no
  Dropout, and no other Opacus-incompatible layers. Don't add them in DP runs
  either — keep the architecture identical so utility differences are
  attributable to DP, not to architecture changes.
- **ε = ∞ baseline.** When running the ε sweep, also run an "ε = ∞" config
  that uses Opacus' wrapper with no noise / no clipping (or just the regular
  baseline) as a control. This isolates the effect of DP from any other
  training-loop differences Opacus introduces.

## Reproducibility checklist

Every training run records, in `runs/{experiment}/seed_{N}/final.json`:

- Full config (model, optimizer, hyperparameters, DP settings)
- Random seed
- Git commit hash
- Wall-clock time
- Final train/eval metrics

If you change the model architecture, optimizer, or split sizes, **bump the
experiment name** so old runs aren't silently overwritten.
