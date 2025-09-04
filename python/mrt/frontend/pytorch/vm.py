import typing
import torch
from collections import namedtuple

from .converter import *
from .types import *

from mrt.mir.symbol import *
from mrt.common.types import *

Executor = namedtuple("Executor", ["vm", "device"])

def create_executor(
        symbol: MultiHeadSymbol, params: ParametersT,
        device: str = "cpu",
        target: str = "",
        ) -> Executor:
    mod = mrt_to_pytorch(symbol, params)
    mod.eval()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return Executor(mod.to(device), device)

def run_executor(
        executor: Executor,
        data: typing.Optional[np.ndarray] = None,
        data_dict: ParametersT = {}) -> OpNumpyT:
    (vm, device) = executor
    for k, v in data_dict.items():
        data_dict[k] = torch.from_numpy(v).to(device)
    if data is not None:
        data = torch.from_numpy(data).to(device)
    out = vm(data, **data_dict)
    return data_to_mrt(out.detach().cpu())

def infer(graph: MultiHeadSymbol, params: ParametersT,
          data: typing.Optional[np.ndarray] = None,
          data_dict: ParametersT = {},
          device: str = "cpu",
          **kwargs):
    executor = create_executor(graph, params, device=device, **kwargs)
    out = run_executor(executor, data, data_dict)
    return out
