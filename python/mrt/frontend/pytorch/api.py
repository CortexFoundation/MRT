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
        }

def create_input_vars(ep: torch.export.ExportedProgram):
    """Create relax input vars."""
    parameters_buffers_constants = OrderedDict()
    user_inputs = OrderedDict()

    for spec in ep.graph_signature.input_specs:
        name_hint = spec.arg.name
        print("process vars: ", spec)
        if spec.kind is torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
            torch_shape = ep.tensor_constants[spec.target].shape
            torch_dtype = ep.tensor_constants[spec.target].dtype
        elif spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
            pass
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
        print(">> vars: ", out)
        if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
            user_inputs[name_hint] = out
        else:
            parameters_buffers_constants[name_hint] = out

    return parameters_buffers_constants, user_inputs

def pytorch_to_mrt(
        ep: torch.export.ExportedProgram,
        model_name: str = "main"
        ) -> typing.Tuple[MultiHeadSymbol, ParametersT]:
    env: typing.Dict[fx.Node, Symbol] = {}

    parameter_buffer_constant_vars, user_input_vars = create_input_vars(ep)
    inputs_vars = user_input_vars.copy()
    inputs_vars.update(parameter_buffer_constant_vars)

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
        if node.op == "placeholder":
            if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                print("ignore")
                # Ignore sym input
                continue

            if node.name not in inputs_vars:
                meta = node.meta["tensor_meta"]
                inputs_vars[node.name] = op.variable(
                        node.name, meta.shape, meta.dtype)
            env[node] = inputs_vars[node.name]
        elif node.op == "output":
            assert len(args) == 1
            if isinstance(args[0], list):
                env[node] = op._new_op(TUPLE, *args[0], **attrs)
            else:
                env[node] = op._new_op(TUPLE, args[0], **attrs)
            output = MultiHeadSymbol({ model_name: env[node] })
        elif node.op == "get_attr":
            env[node] = getattr(ep.graph_module, node.target)
        elif node.op == "call_function":
            func_name = node.target.__name__
            assert (
                    func_name in TORCH_MRT_OP_MAP
                    ), f"Unsupported function type {func_name}"
            op_name = TORCH_MRT_OP_MAP[func_name]
            env[node] = op._new_op(op_name, *args, **attrs)
        else:
            raise ValueError(f"Unsupported op {node.op}")

        print(">> ", env[node])
    assert output is not None

    to_bind_parameters = ChainMap(
        ep.state_dict, OrderedDict(ep.named_buffers()), ep.constants)
    params = {}
    for tensor_name, tensor_value in to_bind_parameters.items():
        # find relax var name from graph signature
        bind_name = None
        for spec in ep.graph_signature.input_specs:
            if tensor_name == spec.target:
                bind_name = spec.arg.name
                break
        if bind_name is not None:
            params[bind_name] = tensor_value.detach().cpu().numpy()
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
                if isinstance(value, np.ndarray):
                    self.register_buffer(name, torch.from_numpy(value))
                else:
                    self.register_buffer(name, torch.tensor(value))
        
        def forward(self, *inputs):
            # Get the main symbol (assuming single output for now)
            main_sym = self.graph.get("main", list(self.graph.values())[0])
            
            # Execute the graph
            return self._execute_symbol(main_sym, list(inputs))
        
        def _execute_symbol(self, symbol: Symbol, inputs: typing.List[torch.Tensor]):
            """Execute a single symbol with given inputs."""
            
            if symbol.op_name == VAR:
                # Variable - either input or parameter
                if symbol.name in self.params:
                    return getattr(self, symbol.name)
                else:
                    # Find input by name or position
                    # For now, assume single input
                    return inputs[0] if inputs else None
            
            elif symbol.op_name == TUPLE:
                # Tuple operation - return tuple of results
                results = []
                for arg in symbol.args:
                    result = self._execute_symbol(arg, inputs)
                    results.append(result)
                return results[0] if len(results) == 1 else tuple(results)
            
            elif symbol.op_name == DENSE:
                # Dense/Linear layer
                assert len(symbol.args) >= 2, f"Dense expects at least 2 args, got {len(symbol.args)}"
                input_tensor = self._execute_symbol(symbol.args[0], inputs)
                weight_tensor = self._execute_symbol(symbol.args[1], inputs)
                
                # Check if bias is provided
                if len(symbol.args) > 2:
                    bias_tensor = self._execute_symbol(symbol.args[2], inputs)
                    return F.linear(input_tensor, weight_tensor, bias_tensor)
                else:
                    return F.linear(input_tensor, weight_tensor)
            
            elif symbol.op_name == RELU:
                # ReLU activation
                assert len(symbol.args) == 1, f"ReLU expects 1 arg, got {len(symbol.args)}"
                input_tensor = self._execute_symbol(symbol.args[0], inputs)
                return F.relu(input_tensor)
            
            else:
                raise NotImplementedError(f"Operation {symbol.op_name} not implemented in mrt_to_pytorch")
    
    return MRTModule(graph, params)

def type_infer(symbol: Symbol) -> Symbol:
    raise RuntimeError("")
