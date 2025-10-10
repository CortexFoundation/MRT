"""
Tests for PyTorch dtype conversion helpers.
"""

from os import path
import sys

# Ensure the project "python" dir is on the path
ROOT = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
sys.path.insert(0, path.join(ROOT, "python"))

import torch
import pytest

from mrt.frontend.pytorch.types import dtype_to_mrt, dtype_from_mrt

# pytest tests/frontend/pytorch/test_types.py -q

@pytest.mark.parametrize("dt", [
    torch.float32, torch.float64, torch.float16, torch.bfloat16,
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.uint8, torch.bool,
    torch.complex64, torch.complex128,
])
def test_dtype_roundtrip(dt):
    s = dtype_to_mrt(dt)
    assert isinstance(s, str)
    rt = dtype_from_mrt(s)
    assert isinstance(rt, torch.dtype)
    assert rt == dt


def test_dtype_invalid():
    with pytest.raises(AttributeError):
        dtype_from_mrt("float128")

