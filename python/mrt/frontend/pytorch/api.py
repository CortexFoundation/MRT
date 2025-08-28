import typing
import torch
from torch import fx
import numpy as np
from collections import ChainMap, OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import sys

from mrt.mir.symbol import Symbol, MultiHeadSymbol, sym2list
from mrt.mir import op
from mrt.mir.opns import *
from mrt.common.types import ParametersT
from mrt.common.utils import N
from .types import convert_to_py, convert_torch_dtype

__all__ = ["pytorch_to_mrt", "mrt_to_pytorch", "from_pytorch", "type_infer"]

TORCH_MRT_OP_MAP = {
        "linear.default": DENSE,
        "relu.default": RELU,
        "relu_.default": RELU,
        "conv2d.default": CONV2D,
        "batch_norm.default": BATCH_NORM,
        "adaptive_avg_pool2d.default": ADAPTIVE_AVG_POOL2D,
        "max_pool2d.default": MAX_POOL2D,
        "flatten.using_ints": FLATTEN,
        "add_.Tensor": ADD,
        }
MRT_TORCH_OP_MAP = {
        TUPLE: lambda *args: [ *args ],

        RELU: F.relu,
        DENSE: F.linear,
        CONV2D: F.conv2d,
        BATCH_NORM: F.batch_norm,

        MAX_POOL2D: F.max_pool2d,
        ADAPTIVE_AVG_POOL2D: F.adaptive_avg_pool2d,

        FLATTEN: torch.flatten,

        ADD: torch.add,
        }

def create_parameters(ep: torch.export.ExportedProgram):
    """Create relax input vars."""
    parameters_buffers_constants = OrderedDict()

    to_bind_parameters = ChainMap(
        ep.state_dict, OrderedDict(ep.named_buffers()), ep.constants)
    params = {}
    #  for tensor_name, tensor_value in to_bind_parameters.items():
    #      # find relax var name from graph signature
    #      bind_name = None
    #      for spec in ep.graph_signature.input_specs:
    #          if tensor_name == spec.target:
    #              print("match bind target:", spec)
    #              bind_name = spec.arg.name
    #              break
    #      if bind_name is not None:
    #          params[bind_name] = tensor_value.detach().cpu().numpy()

    for spec in ep.graph_signature.input_specs:
        name_hint = spec.arg.name
        print("process vars: ", spec)
        if spec.kind is torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
            torch_shape = ep.tensor_constants[spec.target].shape
            torch_dtype = ep.tensor_constants[spec.target].dtype
        elif spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
            continue
        else:
            # PARAMETER or BUFFER
            torch_shape = ep.state_dict[spec.target].shape
            torch_dtype = ep.state_dict[spec.target].dtype


        dshape = [
            str(s) if isinstance(s, torch.SymInt) else s
            for s in torch_shape
        ]
        dtype = convert_torch_dtype(torch_dtype)

        out = op.variable(name_hint, dshape, dtype)
        params[name_hint] = to_bind_parameters[spec.target].detach().numpy().astype(dtype)
        assert dshape == list(params[name_hint].shape)
        print(">> vars: ", out)
        parameters_buffers_constants[name_hint] = out

    return parameters_buffers_constants, params

def pytorch_to_mrt(
        ep: torch.export.ExportedProgram,
        model_name: str = "main"
        ) -> typing.Tuple[MultiHeadSymbol, ParametersT]:
    env: typing.Dict[fx.Node, Symbol] = {}

    param_vars, params = create_parameters(ep)

    def _retrieve_args(node):
        if isinstance(node, fx.Node):
            return env[node]
        elif isinstance(node, (tuple, list)):
            return [_retrieve_args(a) for a in node]
        elif isinstance(node, dict):
            return {_retrieve_args(k): _retrieve_args(v) \
                    for k, v in node.items()}
        elif isinstance(node, (int, float, str, bool)) or node is None:
            return node
        else:
            raise RuntimeError(f"Unsupported argument type: {type(node)} - {node}")

    nodes: typing.List[fx.Node] = ep.graph.nodes
    for node in nodes:
        print("process: ", node)
        args = [_retrieve_args(a) for a in node.args]
        attrs = {}
        func_name = None
        if node.op == "placeholder":
            if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                # Ignore sym input
                print("ignore")
                continue

            if node.name not in param_vars:
                meta = node.meta["tensor_meta"]
                env[node] = op.variable(
                        node.name, meta.shape, meta.dtype)
            else:
                env[node] = param_vars[node.name]
        elif node.op == "output":
            assert len(args) == 1
            env[node] = args[0][0]
            output = MultiHeadSymbol({ model_name: env[node] })
        elif node.op == "get_attr":
            env[node] = getattr(ep.graph_module, node.target)
        elif node.op == "call_function":
            func_name = node.target.__name__
            assert (
                    func_name in TORCH_MRT_OP_MAP
                    ), f"Unsupported function type {func_name}"
            op_name = TORCH_MRT_OP_MAP[func_name]
            
            # Extract attributes for specific operations
            if func_name == "conv2d.default":
                # Conv2D: [input, weight, bias, stride, padding, dilation, groups]
                # bias can be None
                attrs = {
                    'strides': args[3] if len(args) > 3 else [1, 1],
                    'padding': args[4] if len(args) > 4 else [0, 0],
                    'dilation': args[5] if len(args) > 5 else [1, 1],
                    'groups': args[6] if len(args) > 6 else 1
                }
                args = [a for a in args[:3] if a is not None]
            elif func_name == "batch_norm.default":
                # BatchNorm: [input, weight, bias, running_mean, running_var, training, momentum, eps]
                attrs = {
                    'training': False,  # Force training to False
                    'momentum': args[6] if len(args) > 6 else 0.1,
                    'eps': args[7] if len(args) > 7 else 1e-5
                }
                args = [a for a in args[:5] if a is not None]
            elif func_name == "max_pool2d.default":
                attrs = {
                    'kernel_size': args[1] if len(args) > 1 else [1, 1],
                    'strides': args[2] if len(args) > 2 else args[1] if len(args) > 1 else [1, 1],
                    'padding': args[3] if len(args) > 3 else [0, 0]
                }
                args = args[:1]
            elif func_name == "adaptive_avg_pool2d.default":
                attrs = {'output_size': args[1] if len(args) > 1 else (1, 1)}
                args = args[:1]
            elif func_name == "flatten.using_ints":
                attrs = {
                    'start_dim': args[1] if len(args) > 1 else 1,
                    'end_dim': args[2] if len(args) > 2 else -1
                }
                args = args[:1]
            # relu_.default, relu.default, add_.Tensor just pass through args

            env[node] = op._new_op(op_name, *args, **attrs)
        else:
            raise ValueError(f"Unsupported op {node.op}")

        print(">> ", env[node])
    assert output is not None
    return output, params

def mrt_to_pytorch(graph: MultiHeadSymbol, params: ParametersT) -> torch.nn.Module:
    """Convert MRT graph back to PyTorch module."""
    
    class MRTModule(torch.nn.Module):
        def __init__(self, graph: MultiHeadSymbol, params: ParametersT):
            super().__init__()
            self.graph = graph
            self.params = params
            
            # Register parameters as buffers
            for name, value in params.items():
                assert isinstance(value, np.ndarray)
                self.register_buffer(name, torch.from_numpy(value))
        
        def forward(self, data, **data_dict):
            # Get the main symbol (assuming single output for now)
            main_sym = self.graph["main"]

            env: Dict[Symbol, F.Tensor] = {}
            for sym in sym2list(main_sym):
                print("<<", sym)
                if op.is_input(sym, self.params):
                    env[sym] = data_dict.get(sym.name, data)
                elif op.is_param(sym, self.params):
                    env[sym] = getattr(self, sym.name)
                else:
                    args = [env[a] for a in sym.args]
                    attrs = {k: v for k, v in sym.attrs.items()}
                    if sym.op_name in [BATCH_NORM]:
                        # reorder in [input, running_mean, running_var, weight, bias]
                        args = [*args, None, None, None, None][:5]
                        args = [ args[0], args[3], args[4], args[1], args[2] ]
                    if sym.op_name in [CONV2D, MAX_POOL2D]:
                        attrs["stride"] = attrs.pop("strides")
                    env[sym] = MRT_TORCH_OP_MAP[sym.op_name](*args, **attrs)
                print(">>", env[sym].flatten()[:5])
            return env[main_sym]


    return MRTModule(graph, params)

def type_infer(symbol: Symbol) -> Symbol:
    raise RuntimeError("")
