"""
Test script for EfficientNet to MRT conversion.
"""

from os import path
import sys

ROOT = path.dirname(path.dirname(path.dirname(
    path.realpath(__file__)))
)
sys.path.insert(0, path.join(ROOT, "python"))

import torch
import numpy as np
import torchvision

from mrt.frontend.pytorch import pytorch_to_mrt, mrt_to_pytorch

def test_efficientnet_conversion():
    """Test conversion of EfficientNet model."""
    
    # Load pre-trained EfficientNet model
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model.eval()
    
    # Create example inputs
    example_inputs = torch.randn(1, 3, 224, 224)
    
    print("Original EfficientNet PyTorch model:")
    # print(model)
    
    # Convert to MRT
    print("\nConverting EfficientNet PyTorch model to MRT...")
    ep = torch.export.export(model, (example_inputs,))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    
    # Test inference with original model
    with torch.no_grad():
        original_output = model(example_inputs)
    
    # Convert back to PyTorch
    print("\nConverting MRT back to PyTorch...")
    torch_model = mrt_to_pytorch(mrt_graph, mrt_params)
    
    # Test inference with converted model
    with torch.no_grad():
        converted_output = torch_model(example_inputs)
    
    assert np.allclose(original_output.numpy(), converted_output.numpy(), atol=1e-5), "Outputs of original and converted models do not match."
    print("EfficientNet conversion test passed!")

if __name__ == "__main__":
    test_efficientnet_conversion()
