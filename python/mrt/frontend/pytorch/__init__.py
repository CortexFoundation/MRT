"""
PyTorch frontend for MRT (Model Runtime Toolkit).

This module provides functions to convert PyTorch models to MRT format
and vice versa, enabling PyTorch models to benefit from MRT's quantization
and optimization capabilities.
"""

from .api import pytorch_to_mrt, mrt_to_pytorch, type_infer

# Expose the required functions for the frontend API
from_frontend = pytorch_to_mrt
to_frontend = mrt_to_pytorch

__all__ = ["pytorch_to_mrt", "mrt_to_pytorch", "from_frontend", "to_frontend", "type_infer"]
