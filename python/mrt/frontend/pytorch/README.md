# PyTorch Frontend for MRT

This frontend enables converting PyTorch models to MRT format and vice versa, allowing PyTorch models to benefit from MRT's quantization and optimization capabilities.

## Installation

Make sure you have PyTorch installed:
```bash
pip install torch
```

## Usage

### Converting PyTorch to MRT

```python
import torch
from mrt.frontend.pytorch import pytorch_to_mrt

# Define your PyTorch model
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.linear(x))

# Create model and example inputs
model = MyModel()
example_inputs = torch.randn(1, 10)

# Convert to MRT
mrt_graph, mrt_params = pytorch_to_mrt(model, example_inputs, model_name="my_model")
```

### Converting MRT back to PyTorch

```python
from mrt.frontend.pytorch import mrt_to_pytorch

# Convert MRT back to PyTorch
torch_model = mrt_to_pytorch(
    mrt_graph, 
    mrt_params, 
    input_shapes=[[1, 10]], 
    input_dtypes=["float32"]
)
```

### Complete Example

```python
import torch
from mrt.frontend.pytorch import pytorch_to_mrt, mrt_to_pytorch

# 1. Create a PyTorch model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(10, 20)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(20, 5)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel()
example_inputs = torch.randn(1, 10)

# 2. Convert to MRT
mrt_graph, mrt_params = pytorch_to_mrt(model, example_inputs)

# 3. Convert back to PyTorch
reconstructed_model = mrt_to_pytorch(
    mrt_graph, 
    mrt_params, 
    input_shapes=[[1, 10]], 
    input_dtypes=["float32"]
)

# 4. Test both models
model.eval()
with torch.no_grad():
    original_output = model(example_inputs)
    reconstructed_output = reconstructed_model(example_inputs)

print("Original output:", original_output)
print("Reconstructed output:", reconstructed_output)
```

## Supported Operations

The PyTorch frontend currently supports conversion of the following operations:

- Basic arithmetic: `add`, `sub`, `mul`, `div`
- Matrix operations: `matmul`, `linear`
- Activation functions: `relu`, `sigmoid`, `tanh`, `softmax`
- Convolution: `conv2d`
- Pooling: `max_pool2d`, `avg_pool2d`, `adaptive_avg_pool2d`
- Normalization: `batch_norm`
- Shape operations: `flatten`, `reshape`, `transpose`
- Reduction operations: `sum`, `mean`, `max`

## Limitations

- The current implementation is a prototype and may not support all PyTorch operations
- Complex control flow (if/else, loops) is not fully supported
- Some advanced PyTorch features may not convert correctly

## Development

To run tests:
```bash
cd /path/to/MRT
python tests/test.pytorch.py
```

To run the demo:
```bash
cd /path/to/MRT
python python/mrt/frontend/pytorch/demo.py
```