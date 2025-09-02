"""
Test script for ShuffleNetV2 to MRT conversion.
"""

from os import path
import sys

ROOT = path.dirname(path.dirname(path.dirname(
    path.realpath(__file__))))
sys.path.insert(0, path.join(ROOT, "python"))

import torch
import numpy as np
import torchvision

from mrt.mir import helper
from mrt.frontend.pytorch import pytorch_to_mrt, mrt_to_pytorch

def test_shufflenet_conversion():
    """Test conversion of ShuffleNetV2 model."""
    
    # Load pre-trained ShuffleNetV2 model
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    model.eval()
    
    # Create example inputs
    example_inputs = torch.randn(1, 3, 224, 224)
    
    # Convert to MRT
    ep = torch.export.export(model, (example_inputs,))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)

    # with open("/tmp/shufflenet.log", "w") as f:
    #     f.write(helper.format_symbol(mrt_graph["main"], mrt_params))

    # Test inference with original model
    with torch.no_grad():
        original_output = model(example_inputs)
    
    # Convert back to PyTorch
    torch_model = mrt_to_pytorch(mrt_graph, mrt_params)
    
    # Test inference with converted model
    with torch.no_grad():
        converted_output = torch_model(example_inputs)
    
    assert np.allclose(original_output.numpy(), converted_output.numpy(), atol=1e-5), "Outputs of original and converted models do not match."

if __name__ == "__main__":
    test_shufflenet_conversion()
