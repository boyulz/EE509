# Baseline Results

Configuration: SGD (lr=0.01, momentum=0.9), 100 epochs, 10k train / 10k test.

| Seed | Train acc | Eval acc | Gap |
|---|---|---|---|
| 1 | 1.0000 | 0.7366 | 0.2634 |
| 2 | 1.0000 | 0.7495 | 0.2505 |
| 3 | 1.0000 | 0.7426 | 0.2574 |
| 42 | 1.0000 | 0.7479 | 0.2521 |
| **Mean ± std** | **1.0000** | **0.7442 ± 0.0058** | **0.2558 ± 0.0058** |

Note: Our Purchase-100 test accuracy is higher than the original Shokri 2017
report (~0.55-0.65) because we use the public preprocessed version from the
Shokri lab GitHub, which differs slightly from the paper's exact preprocessing.
This matches what later works (BLINDMI, SHAPr) report on the same version.
The generalization gap is sufficient to support meaningful MIA evaluation.