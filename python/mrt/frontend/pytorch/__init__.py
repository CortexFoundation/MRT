"""
PyTorch frontend for MRT (Model Runtime Toolkit).

This module provides functions to convert PyTorch models to MRT format
and vice versa, enabling PyTorch models to benefit from MRT's quantization
and optimization capabilities.
"""

from .converter import pytorch_to_mrt, mrt_to_pytorch, type_infer
from .types import data_to_mrt, data_to_torch
from .vm import create_executor, run_executor, infer

# Expose the required functions for the frontend API
from_frontend = pytorch_to_mrt
to_frontend = mrt_to_pytorch

model_from_frontend = pytorch_to_mrt
model_to_frontend = mrt_to_pytorch

data_from_frontend = data_to_torch
data_to_frontend = data_to_mrt

# __all__ = ["pytorch_to_mrt", "mrt_to_pytorch", "from_frontend", "to_frontend", "type_infer"]
