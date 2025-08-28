import typing

import torch
import numpy as np

from mrt.common.types import *

# PyTorch specific types
PyTorchModule = torch.nn.Module
PyTorchTensor = torch.Tensor

DefConvertFunc = typing.Callable[[typing.Any], typing.Any]

def to_numpy(data: typing.Union[torch.Tensor, list]) -> OpNumpyT:
    return convert_to_py(
            data, log_default_type=False,
            default_convert_func=lambda x: x.detach().cpu().numpy())

def to_tensor(data: OpNumpyT) -> typing.Union[torch.Tensor, list]:
    return [torch.from_numpy(d) for d in data] \
            if isinstance(data, list) else torch.from_numpy(data)

def convert_torch_dtype(dtype: typing.Union[str, torch.dtype]):
    dtype = dtype.lower() if isinstance(dtype, str) else dtype
    if dtype in ["float", "float32", "torch.float32", torch.float32]:
        return "float32"
    elif dtype in ["float16", "torch.float16", torch.float16]:
        return "float16"
    elif dtype in ["int64", "torch.int64", torch.int64]:
        return "int64"
    elif dtype in ["int32", "torch.int32", torch.int32]:
        return "int32"
    elif dtype in ["bool", "torch.bool", torch.bool]:
        return "bool"
    else:
        raise NotImplementedError("input_type {} is not handled yet".format(dtype))


def convert_to_py(value,
                  log_default_type: bool = False,
                  default_convert_func: DefConvertFunc = lambda x: x,
                  support_numpy: bool = True):
    """ PyTorch type to intrinsic py type. """
    # need to pass the kwargs iterately.
    kwargs = {
            "log_default_type": log_default_type,
            "default_convert_func": default_convert_func,
            "support_numpy": support_numpy,
    }
    if isinstance(value, (list, tuple)):
        return [convert_to_py(v, **kwargs) for v in value]
    elif isinstance(value, (torch.Tensor)):
        if support_numpy:
            return value.detach().cpu().numpy()
        else:
            # Return scalar value if it's a 0-d tensor
            if value.dim() == 0:
                return value.item()
            else:
                return default_convert_func(value)
    elif isinstance(value, (torch.nn.Parameter)):
        if support_numpy:
            return value.data.detach().cpu().numpy()
        else:
            return default_convert_func(value)
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif value is None:
        return value
    elif isinstance(value, (torch.dtype)):
        # Convert torch dtype to string representation
        return str(value)
    elif support_numpy and isinstance(value, np.ndarray):
        return value
    elif log_default_type:
        print(">>> unknown type:", type(value))
    return default_convert_func(value)
