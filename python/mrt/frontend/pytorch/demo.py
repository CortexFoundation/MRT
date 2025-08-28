#!/usr/bin/env python3
"""
Demo script showing how to use the PyTorch frontend for MRT.
"""

import torch
import numpy as np

# Add the project root to the path so we can import mrt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mrt.frontend.pytorch import pytorch_to_mrt, mrt_to_pytorch

def demo():
    """Demonstrate PyTorch to MRT conversion."""
    
    print("PyTorch to MRT Conversion Demo")
    print("=" * 40)
    
    # Define a simple PyTorch model
    class DemoModel(torch.nn.Module):
        def __init__(self):
            super(DemoModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu1 = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
            self.relu2 = torch.nn.ReLU()
            self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
            self.flatten = torch.nn.Flatten()
            self.fc = torch.nn.Linear(32 * 4 * 4, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc(x)
            return x
    
    # Create the model and example inputs
    model = DemoModel()
    example_inputs = torch.randn(1, 3, 8, 8)
    
    print("1. Original PyTorch Model:")
    print(model)
    print()
    
    # Test the original model
    model.eval()
    with torch.no_grad():
        original_output = model(example_inputs)
    
    print(f"2. Original Model Output Shape: {original_output.shape}")
    print(f"   Sample Output Values: {original_output.flatten()[:5].tolist()}")
    print()
    
    # Convert PyTorch model to MRT format
    print("3. Converting PyTorch Model to MRT...")
    mrt_graph, mrt_params = pytorch_to_mrt(model, example_inputs, model_name="demo_model")
    
    print(f"   MRT Graph Keys: {list(mrt_graph.keys())}")
    print(f"   Number of Parameters: {len(mrt_params)}")
    print(f"   Sample Parameter Keys: {list(mrt_params.keys())[:3]}")
    print()
    
    # Convert back to PyTorch
    print("4. Converting MRT back to PyTorch...")
    reconstructed_model = mrt_to_pytorch(
        mrt_graph, 
        mrt_params, 
        input_shapes=[[1, 3, 8, 8]], 
        input_dtypes=["float32"]
    )
    
    print("   Reconstructed Model Created Successfully!")
    print()
    
    # Test the reconstructed model
    print("5. Testing Reconstructed Model...")
    with torch.no_grad():
        reconstructed_output = reconstructed_model(example_inputs)
    
    print(f"   Reconstructed Output Shape: {reconstructed_output.shape}")
    print(f"   Sample Output Values: {reconstructed_output.flatten()[:5].tolist()}")
    print()
    
    print("Demo completed successfully!")

if __name__ == "__main__":
    demo()