"""
Test script for ResNet50 PyTorch to MRT conversion.
"""

from os import path
import sys

ROOT = path.dirname(path.dirname(path.dirname(
    path.realpath(__file__))))
sys.path.insert(0, path.join(ROOT, "python"))

import torch
import torchvision.models as models
import numpy as np

from mrt.frontend.pytorch import pytorch_to_mrt, mrt_to_pytorch
from mrt.mir import helper

def test_resnet50_operations():
    """Test what operations ResNet50 uses."""
    
    # Load pre-trained ResNet50
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    
    # Create example input
    example_inputs = torch.randn(1, 3, 224, 224)
    
    print("Exporting ResNet50...")
    ep = torch.export.export(model, (example_inputs,))
    
    print("Operations used in ResNet50:")
    operations = set()
    for node in ep.graph.nodes:
        if node.op == 'call_function':
            func_name = node.target.__name__
            operations.add(func_name)
            print(f"  {func_name}")
    
    print(f"\nTotal unique operations: {len(operations)}")
    print("Operations:", sorted(operations))
    
    return operations

def test_resnet50_conversion():
    """Test conversion with pre-trained ResNet50."""
    
    # Load pre-trained ResNet50
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    
    # Create example input
    example_inputs = torch.randn(1, 3, 224, 224)
    
    print("Original ResNet50 model loaded")
    print(f"Input shape: {example_inputs.shape}")
    
    # Test inference with original model
    with torch.no_grad():
        original_output = model(example_inputs)
    print(f"Original model output shape: {original_output.shape}")
    print(f"Original model output sample: {original_output.flatten()[:5]}")

    with open("/tmp/resnet50.torch", "w") as f:
        f.write(str(model))
    
    # Convert to MRT
    print("\nConverting ResNet50 to MRT...")
    ep = torch.export.export(model, (example_inputs,))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    print(f"MRT conversion successful!")
    print(f"MRT graph keys: {list(mrt_graph.keys())}")
    print(f"MRT params count: {len(mrt_params)}")

    with open("/tmp/resnet50.log", "w") as f:
        f.write(helper.format_symbol(
            mrt_graph["main"], mrt_params))
    
    # Convert back to PyTorch
    print("\nConverting MRT back to PyTorch...")
    torch_model = mrt_to_pytorch(mrt_graph, mrt_params)
    with open("/tmp/resnet50-ex.torch", "w") as f:
        f.write(str(torch_model))
    
    # Test inference with converted model
    with torch.no_grad():
        converted_output = torch_model(example_inputs)
    print(f"Converted model output shape: {converted_output.shape}")
    print(f"Converted model output sample: {converted_output.flatten()[:5]}")
    
    # Check if outputs match
    assert torch.allclose(original_output, converted_output, atol=1e-5)

if __name__ == "__main__":
    print("Testing ResNet50 operations...")
    operations = test_resnet50_operations()
    
    print("\n" + "="*60 + "\n")
    
    print("Testing ResNet50 conversion...")
    test_resnet50_conversion()
