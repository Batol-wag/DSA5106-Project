"""
test_model_equivalence.py

Purpose:
--------
Test whether the full RepVGG model produces nearly identical outputs
before and after deploy conversion.

Objective:
----------
Validate that structural re-parameterization works correctly across
the entire network, not just a single block.

Procedure:
----------
1. Create model in training form
2. Set eval mode
3. Forward pass → save output
4. Convert entire model to deploy
5. Set eval mode again
6. Forward pass → save output
7. Compare outputs

Expected:
---------
Very small numerical difference (~1e-5)
"""

import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.repvgg_net import create_repvgg_small


def test_model_equivalence():
    torch.manual_seed(0)

    # Input
    x = torch.randn(2, 3, 32, 32)

    # Model (training form)
    model = create_repvgg_small(num_classes=10, deploy=False)

    # Eval mode BEFORE forward
    model.eval()
    print("Model training mode before:", model.training)

    # Forward before deploy
    out_before = model(x)

    # Convert entire model
    model.switch_to_deploy()

    # Eval mode again
    model.eval()
    print("Model training mode after:", model.training)

    # Forward after deploy
    out_after = model(x)

    # Compare
    diff = (out_before - out_after).abs()

    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.10f}")
    print(f"Mean difference: {mean_diff:.10f}")

    assert max_diff < 1e-4, f"Model fusion mismatch too large: {max_diff}"


if __name__ == "__main__":
    test_model_equivalence()