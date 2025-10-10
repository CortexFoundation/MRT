import typing

import torch
import numpy as np

from mrt.common.types import *

def dtype_to_mrt(data: torch.dtype) -> str:
    return str(data).replace("torch.", "")

def dtype_from_mrt(data: str) -> torch.dtype:
    return getattr(torch, data)

def data_to_mrt(data: typing.Union[torch.Tensor, list]) -> OpNumpyT:
    def _as_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, torch.nn.Parameter):
            return _as_numpy(data.data)
        elif isinstance(data, torch.dtype):
            return dtype_to_mrt(data)
            #  keys = [ "float", "int", "bool" ]
            #  for k in keys:
            #      if k in str(data):
            #          return k
        elif isinstance(data, torch.SymInt):
            return str(data)
        raise ValueError(data)
    return to_pydata(data, default_convert_func=_as_numpy)

def data_to_torch(data: OpNumpyT) -> typing.Union[torch.Tensor, list]:
    return [torch.from_numpy(d) for d in data] \
            if isinstance(data, list) else torch.from_numpy(data)
