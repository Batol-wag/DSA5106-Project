# DSA5106 Project — RepVGG with Branch-Level Knowledge Distillation

A reproduction and extension of RepVGG on CIFAR-10. The core extension introduces **branch-level knowledge distillation**: supervising the student's individual 3×3, 1×1, and identity branches against a pre-trained teacher's soft outputs, in addition to standard output-level KD.

## Project Structure

```
.
├── RepVgg_project/              # Baseline training & evaluation
│   ├── models/
│   │   ├── repvgg_block.py      # RepVGG block (training / deploy modes)
│   │   ├── repvgg_net.py        # Full RepVGG classifier for CIFAR-10
│   │   └── baselines.py         # PlainCNN and ResNet-18 baselines
│   ├── utils/
│   │   ├── fusion.py            # Conv-BN fusion, re-parameterization helpers
│   │   └── metrics.py           # Accuracy / loss utilities
│   ├── tests/
│   │   ├── test_block_equivalence.py
│   │   └── test_model_equivalence.py
│   ├── train.py                 # Baseline single-model training
│   ├── evaluate.py              # Evaluation & deploy-mode validation
│   ├── run_all.py               # Train all three baselines in sequence
│   └── plot_histories.py        # Plot training curves
├── repvgg_with_branches.py      # RepVGG variant that exposes branch outputs
├── distillation_loss.py         # Branch-level KD loss (CE + output KL + branch KL)
├── train_with_distillation.py   # Distillation training entry point
├── training_utils.py            # Shared augmentation & scheduling utilities
├── metrics_report.py            # Aggregate metrics & markdown summary
├── run_full_pipeline.sh         # End-to-end pipeline script
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training

### 1. Baseline models only

```bash
cd RepVgg_project
python run_all.py
```

Trains RepVGG, PlainCNN, and ResNet-18. Checkpoints and histories are written to `RepVgg_project/checkpoints/` and `RepVgg_project/results/`.

### 2. Full pipeline (baselines + distillation)

```bash
bash run_full_pipeline.sh
```

By default this trains all three baselines, then runs three distillation presets (`default`, `strong_branch`, `light_distill`) using the RepVGG checkpoint as teacher.

**Options:**

```bash
# Use a specific teacher and preset
bash run_full_pipeline.sh --teacher_model resnet18 --distillation_config strong_branch

# Custom hyperparameters
bash run_full_pipeline.sh \
  --teacher_model repvgg \
  --temperature 3.5 \
  --alpha_ce 1.5 \
  --alpha_output_kl 1.0 \
  --alpha_branch_kl 0.75
```

### 3. Distillation training only

```bash
python train_with_distillation.py \
  --teacher_model repvgg \
  --teacher_checkpoint RepVgg_project/checkpoints/repvgg_best.pth \
  --distillation_config default
```

If `--teacher_checkpoint` is omitted the script resolves it automatically from `RepVgg_project/checkpoints/<teacher_model>_best.pth`.

## Distillation Loss

`distillation_loss.py` implements `BranchLevelDistillationLoss`:

| Term | Description |
|------|-------------|
| CE loss | Standard cross-entropy against ground-truth labels |
| Output KL | KL divergence between student and teacher final logits |
| Branch KL | KL divergence between each branch output (3×3, 1×1, identity) and teacher logits |

**Distillation presets:**

| Config | temperature | alpha_ce | alpha_output_kl | alpha_branch_kl |
|--------|-------------|----------|-----------------|-----------------|
| `default` | 4.0 | 1.0 | 2.0 | 1.0 |
| `strong_branch` | 5.0 | 1.0 | 1.5 | 2.0 |
| `light_distill` | 3.0 | 2.0 | 1.0 | 0.5 |

## Training Defaults

All training (baseline and distillation) uses:

- Epochs: 30
- Batch size: 128
- Learning rate: 0.1 with `CosineAnnealingLR`
- Augmentation: Cutout, Mixup (α=0.2), label smoothing=0.1

## Output Locations

| Source | Checkpoints | Histories |
|--------|-------------|-----------|
| Baseline (`run_all.py`) | `RepVgg_project/checkpoints/` | `RepVgg_project/results/` |
| Distillation (`train_with_distillation.py`) | `checkpoints/` | `results/` |

Distillation files follow the pattern `repvgg_distilled_<config>_{best,last}.pth`.

## Metrics Report

```bash
cd RepVgg_project
python ../metrics_report.py
```

Outputs a JSON and Markdown summary of model sizes, parameter counts, test accuracies, and deploy-mode accuracy gap.

## Tests

```bash
cd RepVgg_project
pytest tests/
```

Tests verify that training-mode and deploy-mode outputs are numerically equivalent after re-parameterization.
