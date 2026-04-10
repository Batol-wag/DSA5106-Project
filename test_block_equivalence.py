import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.repvgg_block import RepVGGBlock

def test_block_equivalence():
    torch.manual_seed(0)

    # Create input
    x = torch.randn(2, 32, 32, 32)

    # Create block
    block = RepVGGBlock(32, 32, stride=1, deploy=False)
    block.eval()  # VERY IMPORTANT

    # Output before deploy
    out_train = block(x)

    # Convert to deploy
    block.switch_to_deploy()
    block.eval()

    # Output after deploy
    out_deploy = block(x)

    # Compare
    diff = (out_train - out_deploy).abs()

    print("Max difference:", diff.max().item())
    print("Mean difference:", diff.mean().item())

    assert diff.max() < 1e-4, "Fusion mismatch too large!"


if __name__ == "__main__":
    test_block_equivalence()