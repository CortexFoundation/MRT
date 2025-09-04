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
from collections import namedtuple

from mrt.frontend.pytorch import pytorch_to_mrt, mrt_to_pytorch, type_infer
from mrt.frontend.pytorch import vm
from mrt.mir import helper, symbol as sx

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
    
    # Test inference with original model
    with torch.no_grad():
        original_output = model(example_inputs)
    print(f"Original model output shape: {original_output.shape}")
    print(f"Original model output sample: {original_output.flatten()[:5]}")
    
    # Convert to MRT
    print("\nConverting ResNet50 to MRT...")
    ep = torch.export.export(model, (example_inputs,))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)

    with open("/tmp/resnet50.log", "w") as f:
        f.write(helper.format_symbol(
            mrt_graph["main"], mrt_params))
    
    # Convert back to PyTorch
    print("\nConverting MRT back to PyTorch...")
    torch_model = mrt_to_pytorch(mrt_graph, mrt_params)
    ep = torch.export.export(model, (example_inputs,))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    torch_model = mrt_to_pytorch(mrt_graph, mrt_params)

    # with open("/tmp/resnet50-ex.torch", "w") as f:
    #     f.write(str(torch_model))
    
    # Test inference with converted model
    with torch.no_grad():
        converted_output = torch_model(example_inputs)
    print(f"Converted model output shape: {converted_output.shape}")
    print(f"Converted model output sample: {converted_output.flatten()[:5]}")
    
    # Check if outputs match
    assert torch.allclose(original_output, converted_output, atol=1e-5)


def test_resnet50_infer():
    """Test conversion with pre-trained ResNet50."""
    
    # Load pre-trained ResNet50
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    example_inputs = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        original_output = model(example_inputs).numpy()

    print("\nConverting ResNet50 to MRT...")
    ep = torch.export.export(model, (example_inputs,))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)

    print("\nTesting MRT Infer API...")
    out = vm.infer(mrt_graph, mrt_params, example_inputs.numpy())
    assert np.allclose(original_output, out, atol=1e-5)

    out = vm.infer(mrt_graph, mrt_params, example_inputs.numpy(), device="cuda:0")
    #  print(original_output.flatten()[:10], out.flatten()[:10])
    assert np.allclose(original_output, out, atol=0.01)

    out = vm.infer(mrt_graph, mrt_params, data_dict={"x": example_inputs.numpy()}, device="cuda:0")
    assert np.allclose(original_output, out, atol=0.01)

def test_resnet50_type_infer():
    """Test type inference with ResNet50."""
    
    # Load pre-trained ResNet50
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    
    # Create example input
    example_inputs = torch.randn(1, 3, 224, 224)
    
    print("\nConverting ResNet50 to MRT for type inference test...")
    ep = torch.export.export(model, (example_inputs,))
    graph, params = pytorch_to_mrt(ep)

    #  model = mrt_to_pytorch(graph, params)
    #  ep = torch.export.export(model, (example_inputs,))
    #  graph, params = pytorch_to_mrt(ep)
    
    main_symbol = graph["main"]
    
    print("\nTesting type inference for ResNet50...")

    #  # Test type inference
    inferred_symbol = type_infer(main_symbol)

    for sym in sx.sym2list(inferred_symbol):
        assert sym.shape is not None, sym
        assert sym.dtype is not None, sym

    #  # Verify the shape is correct
    assert tuple(inferred_symbol.shape) == (1, 1000), f"Expected shape (1, 1000), but got {inferred_symbol.shape}"


    print("ResNet50 type inference test passed!")

if __name__ == "__main__":
    #  print("Testing ResNet50 operations...")
    #  operations = test_resnet50_operations()

    #  print("\n" + "="*60 + "\n")

    #  print("Testing ResNet50 conversion...")
    #  test_resnet50_conversion()

    print("\n" + "="*60 + "\n")

    test_resnet50_infer()
    #  test_resnet50_type_infer()
