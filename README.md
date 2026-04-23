# DSA5106 Project — RepVGG: Evaluation, Reproduction, and Extension of VGG-Style ConvNets

A reproduction and extension of **RepVGG on CIFAR-10**.  

The project consists of:

- **Reproduction:** Reproducing RepVGG architecture and validating structural re-parameterization
- **Extension:** Branch-Level Knowledge Distillation, where the student's internal 3×3, 1×1, and identity branches are supervised using teacher soft targets in addition to standard output-level distillation.

Detailed experiments, analysis, and results are documented in the final project report.

---

## Repository Organization

This repository is organized into multiple branches to separate reproduction and extension work completed in parallel.

| Branch | Purpose |
|---------|----------|
| `main` | Core RepVGG reproduction pipeline |
| `extension-teacher-model` | Teacher model implementation and training |
| `extension-inference-testing` | Additional inference/testing experiments |
| `extension-branch-distillation` | Branch-level knowledge distillation extension |

The `main` branch should be treated as the primary branch for reviewing the reproduction work.

---

## Main Branch Structure (`main`)

```bash
.
├── RepVgg_project/
│   ├── figures/                        # Training plots and visual outputs
│   ├── models/
│   │   ├── repvgg_block.py             # RepVGG block (training/deploy modes)
│   │   ├── repvgg_net.py               # Full RepVGG architecture
│   │   └── baselines.py                # PlainCNN and ResNet baselines
│   ├── utils/
│   │   ├── fusion.py                   # Conv-BN fusion utilities
│   │   └── metrics.py                  # Accuracy/loss utilities
│   ├── tests/
│   │   ├── test_block_equivalence.py
│   │   └── test_model_equivalence.py
│   ├── train.py                        # Train individual model
│   ├── run_all.py                      # Run full reproduction experiments
│   ├── evaluate.py                     # Model evaluation
│   ├── compare_models.py               # Compare baseline performances
│   ├── deploy_equivalence_eval.py      # Validate deploy equivalence
│   └── plot_histories.py               # Generate training plots
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproduction Pipeline (`main` branch)

### 1. Run full reproduction experiments

This trains all RepVGG variants and baseline models used in the reproduction study.

```bash
cd RepVgg_project
python run_all.py
```

---

### 2. Generate training plots

After training completes:

```bash
python plot_histories.py
```

---

### 3. Validate structural re-parameterization

This verifies that model performance remains consistent before and after deployment conversion.

```bash
python deploy_equivalence_eval.py
```

---

## Optional Analysis Scripts

### Evaluate trained models

```bash
python evaluate.py
```

### Compare model performance

```bash
python compare_models.py
```

### Train a single model manually

```bash
python train.py
```

---

## Extension Branches

### `extension-teacher-model`

Contains:

- Teacher model training
- Architecture modifications
- Teacher checkpoint generation

---

### `extension-inference-testing`

Contains:

- Additional inference experiments
- Deployment testing scripts

---

### `extension-branch-distillation`

Contains:

- Standard output-level knowledge distillation
- Branch-level supervision for:
  - 3×3 branch
  - 1×1 branch
  - identity branch

---

## Tests

```bash
cd RepVgg_project
pytest tests/
```

These tests verify that training-mode and deploy-mode outputs remain numerically equivalent after structural re-parameterization.

---

## Notes

- Branches were intentionally separated to avoid merge conflicts during parallel team development.
- The `main` branch serves as the stable reproduction implementation.
- Full experimental analysis and final results are included in the project report.
