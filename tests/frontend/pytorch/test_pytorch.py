"""
Test script for PyTorch to MRT conversion.
"""

from os import path
import sys

ROOT = path.dirname(path.dirname(path.dirname(
    path.realpath(__file__))))
sys.path.insert(0, path.join(ROOT, "python"))

import torch
import numpy as np

from mrt.frontend.pytorch import pytorch_to_mrt, mrt_to_pytorch, type_infer
from mrt.mir.symbol import Symbol

def test_simple_model():
    """Test conversion with a simple model."""
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear1 = torch.nn.Linear(10, 20)
            self.relu = torch.nn.ReLU()
            torch.nn.BatchNorm2d
            self.linear2 = torch.nn.Linear(20, 5)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    # Create model and example inputs
    model = SimpleModel()
    example_inputs = torch.randn(1, 10)
    
    print("Original PyTorch model:")
    print(model)
    
    # Convert to MRT
    print("\nConverting PyTorch model to MRT...")
    ep = torch.export.export(model, ( example_inputs, ))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    
    # Test inference with original model
    model.eval()
    with torch.no_grad():
        original_output = model(example_inputs)
    
    # Convert back to PyTorch
    print("\nConverting MRT back to PyTorch...")
    torch_model = mrt_to_pytorch(mrt_graph, mrt_params)
    
    # Test inference with converted model
    with torch.no_grad():
        converted_output = torch_model(example_inputs)
    assert np.allclose(original_output.numpy(), converted_output.numpy())

def test_conv_model():
    """Test conversion with a convolutional model."""
    
    class ConvModel(torch.nn.Module):
        def __init__(self):
            super(ConvModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 32, 3, bias=False, padding=2)
            self.pool = torch.nn.AdaptiveAvgPool2d((8, 8))
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(32 * 8 * 8, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x
    
    # Create model and example inputs
    model = ConvModel()
    example_inputs = torch.randn(1, 3, 32, 32)
    
    print("Original Conv PyTorch model:")
    print(model)
    
    # Convert to MRT
    print("\nConverting Conv PyTorch model to MRT...")
    ep = torch.export.export(model, ( example_inputs, ))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    
    # Test inference with original model
    model.eval()
    with torch.no_grad():
        original_output = model(example_inputs)

    # Convert back to PyTorch
    print("\nConverting MRT back to PyTorch...")
    torch_model = mrt_to_pytorch(mrt_graph, mrt_params)
    ep = torch.export.export(model, ( example_inputs, ))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    torch_model = mrt_to_pytorch(mrt_graph, mrt_params)
    
    # Test inference with converted model
    with torch.no_grad():
        converted_output = torch_model(example_inputs)
    assert np.allclose(converted_output.numpy(), original_output.numpy())

def test_type_infer():
    """Test the type inference function."""
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = torch.nn.Linear(10, 5)
            self.relu = torch.nn.ReLU()
            
        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return x
    
    # Create model and example inputs
    model = SimpleModel()
    example_inputs = torch.randn(1, 10)
    
    # Convert to MRT
    ep = torch.export.export(model, ( example_inputs, ))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    
    # Get the main symbol
    main_symbol = mrt_graph["main"]
    
    print("\nTesting type inference...")
    print(f"Main symbol: {main_symbol}")
    print(f"Main symbol shape: {main_symbol.shape}")
    print(f"Main symbol dtype: {main_symbol.dtype}")
    
    # Test type inference
    inferred_symbol = type_infer(main_symbol)
    
    print(f"Inferred symbol: {inferred_symbol}")
    print(f"Inferred symbol shape: {inferred_symbol.shape}")
    print(f"Inferred symbol dtype: {inferred_symbol.dtype}")
    
    # Verify that the symbol has shape and dtype information
    assert inferred_symbol.shape is not None, "Inferred symbol should have shape information"
    assert inferred_symbol.dtype is not None, "Inferred symbol should have dtype information"
    
    print("Type inference test passed!")

if __name__ == "__main__":
    print("Testing PyTorch to MRT conversion...")
    test_simple_model()
    print("\n" + "="*50 + "\n")
    test_conv_model()  # Conv operations now supported
    # print("\n" + "="*50 + "\n")
    # test_type_infer()  # Type inference not yet implemented
