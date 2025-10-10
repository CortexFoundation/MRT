import typing
import torch
from torch import fx
import numpy as np
from collections import ChainMap, OrderedDict
from collections import namedtuple
from dataclasses import dataclass, field
import torch.nn as nn
import torch.nn.functional as F
import sys

from mrt.mir.symbol import Symbol, MultiHeadSymbol, sym2list, transform
from mrt.mir import op
from mrt.mir.opns import *
from mrt.common.types import ParametersT
from mrt.common.utils import N
from .types import data_to_mrt, data_to_torch, dtype_from_mrt

__all__ = ["pytorch_to_mrt", "mrt_to_pytorch", "type_infer"]

Attr = namedtuple('Attr', [ 'name', 'default' ])

@dataclass
class _T:
    op_name: str
    arg_size: int
    attrs: typing.List[Attr] = field(default_factory=list)


TORCH_MRT_OP_MAP = {
        "linear.default": _T(DENSE, 3),
        "conv2d.default": _T(CONV2D, 3, [
            Attr("strides", (1,1)), Attr("padding", (0,0)),
            Attr("dilation", (1, 1)), Attr("groups", 1)]),
        "batch_norm.default": _T(BATCH_NORM, 5, [
            Attr("training", False), Attr("momentum", 0.1), Attr("eps", 1e-5) ]),
        "_native_batch_norm_legit_no_training.default": _T(BATCH_NORM, 5, [
            Attr("momentum", 0.1), Attr("eps", 1e-5) ]),

        "relu.default": _T(RELU, 1),
        "relu_.default": _T(RELU, 1),
        "hardtanh_.default": _T(HARDTANH, 1, [ Attr("min_val", 0.0), Attr("max_val", 6.0) ]),
        "silu_.default": _T(SILU, 1),

        "sigmoid.default": _T(SIGMOID, 1),

        "adaptive_avg_pool2d.default": _T(ADAPTIVE_AVG_POOL2D, 1, [ Attr("output_size", (1,1)) ]),
        "max_pool2d.default": _T(MAX_POOL2D, 1, [
            Attr("kernel_size", (1,1)), Attr("strides", (1,1)), Attr("padding", (0,0)) ]),
        "mean.dim": _T(MEAN, 1, [ Attr("dim", None), Attr("keepdim", False) ]),

        "add.Tensor": _T(ADD, 2), "add_.Tensor": _T(ADD, 2),
        "mul.Tensor": _T(MUL, 2),

        "flatten.using_ints": _T(FLATTEN, 1, [ Attr("start_dim", 1), Attr("end_dim", -1) ]),
        "dropout.default": _T(DROP_OUT, 1, [
            Attr("p", 0.5), Attr("training", True), Attr("inplace", False), ]),
        "dropout_.default": _T(DROP_OUT, 1, [
            Attr("p", 0.5), Attr("training", True), Attr("inplace", False), ]),
        "cat.default": _T(CONCAT, 1, [ Attr("dim", 0) ]),
        "view.default": _T(RESHAPE, 1, [ Attr("shape", ()) ]),
        "transpose.int": _T(TRANSPOSE, 1, [ Attr("dim0", 0), Attr("dim1", 0) ]),
        "contiguous.default": _T(PASS, 1),

        "chunk.default": _T(SPLIT, 1, [ Attr("chunks", 1), Attr("dim", 0) ]),
        "getitem": _T(TUPLE_GET_ITEM, 1, [ Attr("index", 0) ]),
        }

from ..func_mapper import map_function, FunctionMapper

MRT_TORCH_OP_MAP = FunctionMapper({
        TUPLE: lambda *args: [ *args ],
        TUPLE_GET_ITEM: lambda x, index: x[index],

        RELU: F.relu,
        HARDTANH: F.hardtanh,
        SILU: F.silu,
        SIGMOID: F.sigmoid,
        DENSE: F.linear,
        **map_function(
            func_map={
                CONV2D: F.conv2d,
                MAX_POOL2D: F.max_pool2d, },
            attr_map={ "strides": "stride" },
            ),

        **map_function(
            func_map={ SUM: torch.sum, },
            attr_map={ "axis": "dim" },
            ),
        ADAPTIVE_AVG_POOL2D: F.adaptive_avg_pool2d,

        FLATTEN: torch.flatten,
        CLIP: torch.clip,
        AS_TYPE: lambda x, dtype: x.to(
            dtype=dtype_from_mrt(dtype)),
        **map_function(
            func_map={ CONCAT: torch.cat, },
            arg_map=lambda args: [ args, ],
            ),
        RESHAPE: torch.reshape,
        TRANSPOSE: torch.transpose,
        PASS: lambda x: x,
        SPLIT: torch.chunk,

        ADD: torch.add,
        MUL: torch.mul,
        DROP_OUT: F.dropout,
        MEAN: torch.mean,
        })

@MRT_TORCH_OP_MAP.add_arg_mapper({ BATCH_NORM: F.batch_norm })
def _torch_batch_norm_args_reorder(args):
    args = [ *args, None, None, None, None ][:5]
    # reorder in [input, running_mean, running_var, weight, bias]
    return [ args[0], args[3], args[4], args[1], args[2] ]

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
        #  print("process vars: ", spec)
        if spec.kind is torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
            torch_shape = ep.tensor_constants[spec.target].shape
            torch_dtype = ep.tensor_constants[spec.target].dtype
        elif spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
            continue
        else:
            # PARAMETER or BUFFER
            torch_shape = ep.state_dict[spec.target].shape
            torch_dtype = ep.state_dict[spec.target].dtype

        dshape = data_to_mrt(torch_shape)
        dtype = data_to_mrt(torch_dtype)

        out = op.variable(name_hint, dshape, dtype)
        params[name_hint] = to_bind_parameters[spec.target].detach().numpy().astype(dtype)
        assert dshape == list(params[name_hint].shape)
        #  print(">> vars: ", out)
        parameters_buffers_constants[name_hint] = out

    return parameters_buffers_constants, params

def pytorch_to_mrt(
        ep: torch.export.ExportedProgram,
        func_names: typing.List[str] = [ "main", ]
        ) -> typing.Tuple[MultiHeadSymbol, ParametersT]:
    env: typing.Dict[torch.Node, Symbol] = {}

    assert isinstance(ep, torch.export.ExportedProgram), f"input not torch ExportedProgram, but {type(ep)}"
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
        #  print("process: ", node.name, [a for a in node.args])
        shape, dtype = None, None
        if "tensor_meta" in node.meta:
            meta_data = node.meta["tensor_meta"]
            shape = data_to_mrt(meta_data.shape)
            dtype = data_to_mrt(meta_data.dtype)
        #  else:
        #      print(node.name, "has no tensor meta")

        args = [_retrieve_args(a) for a in node.args]
        attrs = {}
        func_name = None
        if node.op == "placeholder":
            if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                # Ignore sym input
                print("ignore")
                continue

            if node.name not in param_vars: # input
                env[node] = op.variable(node.name, shape, dtype)
            else:
                env[node] = param_vars[node.name]
        elif node.op == "output": # [[ out1, out2, out3 ]]
            assert len(args) == 1
            env[node] = args[0]
            assert len(func_names) == len(args[0])
            output = MultiHeadSymbol(zip(func_names, args[0]))
        elif node.op == "get_attr":
            env[node] = getattr(ep.graph_module, node.target)
        elif node.op == "call_function":
            func_name = node.target.__name__
            assert (
                    func_name in TORCH_MRT_OP_MAP
                    ), f"Unsupported function type {func_name}"

            mapper: _T = TORCH_MRT_OP_MAP[func_name]

            attrs = {a.name: args[i+mapper.arg_size] \
                    if len(args) > (i+mapper.arg_size) else a.default \
                    for i, a in enumerate(mapper.attrs)}

            # update args and strip None (optional args must be last)
            args = [a for a in args[:mapper.arg_size] if a is not None]
            if mapper.op_name == CONCAT:
                args = args[0]

            if mapper.op_name == TUPLE_GET_ITEM and args[0].op_name == BATCH_NORM:
                out = args[0]
            else:
                out = op._new_op(
                        mapper.op_name, *args,
                        name=node.name, extra_attrs={ "shape": shape, "dtype": dtype },
                        **attrs)
            env[node] = out
        else:
            raise ValueError(f"Unsupported op {node.op}")

        #  if node.op != "output":
        #      assert isinstance(env[node], Symbol), env[node]
        # print(">> ", env[node])
    assert output is not None
    return output, params

def mrt_to_pytorch(graph: MultiHeadSymbol, params: ParametersT) -> torch.nn.Module:
    """Convert MRT graph back to PyTorch module."""

    class MRTModule(torch.nn.Module):
        def __init__(self, graph: MultiHeadSymbol, params: ParametersT):
            super().__init__()
            self.graph = graph
            self.params = params
            assert len(self.graph) == 1
            self.sym_list = sym2list(graph["main"])

            #  for sym in self.sym_list:
            #      assert sym.op_name in MRT_TORCH_MOD_MAP, sym
            #      setattr(self, sym.name, MRT_TORCH_MOD_MAP[sym.op_name](**sym.attrs))

            # Register parameters as buffers
            for name, value in params.items():
                assert isinstance(value, np.ndarray)
                #  self.register_buffer(name, None)
                self.register_parameter(name, torch.nn.Parameter(
                    torch.from_numpy(value), requires_grad=False))

        def forward(self, data, **data_dict):
            # Get the main symbol (assuming single output for now)
            # TODO: return all graph output
            env: Dict[str, F.Tensor] = {}
            for sym in self.sym_list:
                sn = sym.name
                #  print("<<", sym)
                if op.is_input(sym, self.params):
                    env[sn] = data_dict.get(sn, data)
                elif op.is_param(sym, self.params):
                    env[sn] = getattr(self, sn)
                else:
                    env[sn] = _infer_single_op(sym, env)
                #  print(">>", env[sn].size(), env[sn].dtype, env[sn].flatten()[:5])
            return env[self.graph["main"].name]

    torch_model = MRTModule(graph, params)
    #  print(torch_model.state_dict().keys())
    #  torch_model.load_state_dict({k: torch.from_numpy(v) for k, v in params.items()})
    return torch_model

def _infer_single_op(sym: Symbol, env: typing.Dict[str, F.Tensor]) -> F.Tensor:
    assert op.is_operator(sym), sym

    args = [env[a.name] for a in sym.args]
    attrs = {k: v for k, v in sym.attrs.items()}
    out = MRT_TORCH_OP_MAP[sym.op_name](*args, **attrs)
    return out

def type_infer(symbol: Symbol) -> Symbol:
    """Infer shape and dtype for all symbols in the graph.
    """
    env: Dict[str, F.Tensor] = {}

    def _infer_type(sym: Symbol):
        if op.is_variable(sym):
            out = torch.randn(sym.shape)
        else:
            out = _infer_single_op(sym, env)
            if sym.shape is None:
                sym.shape = data_to_mrt(out.shape)
                sym.dtype = data_to_mrt(out.dtype)
            else:
                assert sym.shape == data_to_mrt(out.shape), f"{sym.shape} vs. {out.shape}"
                assert sym.dtype == data_to_mrt(out.dtype), f"{sym.dtype} vs. {out.dtype}"
        env[sym.name] = out

    out = transform(symbol, _infer_type)
    return out
