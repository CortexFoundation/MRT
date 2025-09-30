from __future__ import annotations
import typing

from functools import wraps
from dataclasses import dataclass

from mrt.common import config
#from mrt.runtime import inference
from mrt.common.utils import *
from mrt.common.types import *

from . import op, opns, opclass
from . import symbol as _symbol


# mrt op visits
@dataclass
class SimplePass:
    symbol: _symbol.Symbol

    """op-level visit of graph
    infer different visit function with different op_name
    return: head symbol processed
    """
    def graph_visits(self) -> _symbol.Symbol:
        env: typing.Dict[str, _symbol.Symbol] = {}
        for sym in _symbol.sym2list(self.symbol):
            assert sym.name not in env, f'{sym.name} NotIn env!'

            # Updating args as passed symbol in env_dict
            sym = sym.copy(args = [env[arg_sym.name] for arg_sym in sym.args])
            assert isinstance(sym, _symbol.Symbol), sym
            out = getattr(self, f"visit_{opns.Opname2Funcname(sym.op_name)}")(sym)
            out = out or sym
            assert isinstance(out, _symbol.Symbol), out
            env[sym.name] = out
        return env[self.symbol.name]

    def _default_visit_op(self, op: _symbol.Symbol) -> _symbol.Symbol:
        return op

    """custom visit of graph
    calling custom_func for all op_name
    return: head symbol processed
    """
    def custom_visits(self, custom_run: _symbol._TransformerParamT, name: str = "", once: bool = False) -> _symbol.Symbol:
        with N(name):
            if once:
                return custom_run(self.symbol)
            return _symbol.transform(self.symbol, custom_run)


# mrt op visits with params, variables
@dataclass
class InferPass(SimplePass):
    params: ParametersT

    def is_input(self, op_: _symbol.Symbol) -> bool:
        return op.is_input(op_, self.params)
    def is_variable(self, op_: _symbol.Symbol) -> bool:
        return op.is_variable(op_, self.params)
    def is_operator(self, op_: _symbol.Symbol) -> bool:
        return op.is_operator(op_, self.params)
    def is_param(self, op_: _symbol.Symbol) -> bool:
        return op_.op_name == opns.VAR and op_.name in self.params

    def get_param(self, op_: _symbol.Symbol) -> OpNumpyT:
        return self.params[op_.name] if self.is_param(op_) else []
    def get_as_numpy(self, op_: _symbol.Symbol) -> OpNumpyT:
        assert self.is_param(op_), f"{op_.name} is not parameter."
        data = self.params[op_.name]
        assert isinstance(data, (tuple, list, np.ndarray)), \
                f"param:{op_.name} not OpNumpyT, get {type(data)}"
        return data

    """custom visit of graph
    calling custom_func for all op_name
    according to how custom_run implemented, params is from argument or class_property
    return: head symbol processed
    """
    def custom_visits_with_params(self, custom_run: _symbol._TransformerParamT, name: str = "", once: bool = False) -> _symbol.Symbol:
        with N(name):
            if once:
                return custom_run(self.symbol, self.params)
            return _symbol.transform(self.symbol, custom_run, params=self.params)

    # From original quantization.Transformer
    def as_parameter(self, data: OpNumpyT, name:str, dtype):
        def _f(data, dtype):
            if isinstance(data, list):
                assert len(data) == len(dtype)
                return [_f(d, t) for d, t in zip(data, dtype)]
            assert isinstance(data, np.ndarray), type(data)
            return data.astype(dtype)
        array = _f(data, dtype)
        shape = np.array(array).shape
        self.params[name] = array
        return opclass.var(array, shape=shape, dtype=dtype)

    def from_np_data(self, sym:_symbol.Symbol, data: np.ndarray, dtype, prefix=None) -> _symbol.Symbol:
        name = N.n(prefix=prefix)
        # some data is np.float/int type, use np.array to wrap it.
        data = np.array(data)
        self.params[name] = data.astype(dtype)
        return opclass.var(name, shape=data.shape, dtype=dtype).like(sym)

    def from_const_data(self, sym:_symbol.Symbol, data: typing.Union[int, float], dtype) -> _symbol.Symbol:
        return self.from_np_data(sym, data, dtype)


# Register MRT all op's default_visit_op function
for op_name in opclass.MRT_OP_MAP.keys():
    funcSuffix = opns.Opname2Funcname(op_name)
    setattr(SimplePass, f"visit_{funcSuffix}", SimplePass._default_visit_op)
    #print(f"visit_, {op_name} => {funcSuffix}", getattr(SimplePass, f"visit_{funcSuffix}"))


# mrt symbol simple pass
class FuseDropoutPass(SimplePass):
    def visit_nn_dropout(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        # make sure op fit again
        if sym.op_name == opns.DROP_OUT:
            return sym.args[0]
        return sym


class FuseTupleGetItemPass(SimplePass):
    def visit_TupleGetItem(self, sym: opclass.TupleGetItem) -> _symbol.Symbol:
        #if sym.op_name == opns.TUPLE_GET_ITEM:
        #    assert sym.index == 0
        #    return sym.args[0]
        return sym


class FuseNaiveSoftmaxPass(SimplePass):
    def visit_nn_softmax(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.SOFTMAX:
            return sym.args[0]
        return sym

    def visit_nn_log_softmax(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.LOG_SOFTMAX:
            return sym.args[0]
        return sym


class FuseMeanPass(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            if sym.op_name == opns.MEAN:
                X = sym.args[0]
                out = opclass.Sum(X, **sym.attrs).like(sym)
                scale = self.from_np_data(sym, np.array(
                    1. * product(out.shape) / product(X.shape)), dtype=out.dtype)
                out = opclass.mul(out, scale)
                return out
            return sym
        return custom_run


class FuseConstantPass(InferPass):
    threshold: typing.ClassVar[float] = 1e-5

    def np_is_zero(self, data) -> float:
        return np.abs(data).max() < self.threshold

    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            if self.is_operator(sym) and all([self.is_param(arg) for arg in sym.args]):
                data = inference.run_single_params(
                        sym, [self.get_as_numpy(a) for a in sym.args])
                return self.as_parameter(data, name=sym.name, dtype=sym.dtype)
            elif sym.is_op(opns.ADD, opns.SUB): # , BIAS_ADD):
                strips = []
                for arg in sym.args:
                    if self.is_param(arg) and self.np_is_zero(self.get_as_numpy(arg)):
                        strips.append(arg)
                args = [a for a in sym.args if a not in strips]
                if len(args) == 1:
                    return args[0]
            elif sym.is_op(opns.SLICE_LIKE):
                if not self.is_param(sym.args[0]):
                    return sym
                a, b = sym.args
                data = inference.run_single_params(
                    sym, [self.get_as_numpy(a), np.zeros(b.shape, b.dtype)])
                return self.as_parameter(data, name=sym.name, dtype=sym.dtype)
            elif sym.is_op(opns.REQUANT):
                if sym.rescale == 1:
                    return sym.args[0]
            elif sym.is_op(opns.ZEROS_LIKE, opns.ONES_LIKE):
                data = inference.run_single_params(sym, [])
                return self.as_parameter(data, name=sym.name, dtype=sym.dtype)
            return sym
        return custom_run


class FuseBatchNormPass(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: opclass.BatchNorm, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            if sym.op_name == opns.BATCH_NORM:
                X, Gamma, Beta, Mean, Var = sym.args
                Gamma = self.get_param(Gamma)
                Beta = self.get_param(Beta)
                Mean = self.get_param(Mean)
                Var = self.get_param(Var)

                assert sym.axis == 1
                Beta = Beta if sym.center else 0
                Gamma = Gamma if sym.scale else 1

                # (x - mean) / sqrt(var + epsilon) * gamma + beta
                Gamma = Gamma / np.sqrt(Var + sym.epsilon)
                # (x - mean) * gamma + beta
                # x * gamma + (beta - mean * gamma)
                bias: np.ndarray = (Beta - Mean * Gamma)
                K = Gamma.shape[0]

                if X.is_op(opns.CONV2D):
                    A, W = X.args
                    assert X.kernel_layout == "OIHW"
                    assert W.shape[0] == K
                    # (A * W) * gamma + bias
                    # A * (W * gamma) + bias
                    W_data = self.get_as_numpy(W) * Gamma.reshape(K, 1, 1, 1)
                    W_sym = self.from_np_data(W, W_data, W.dtype)
                    out = op.nn_conv2d(A, W_sym, **X.attrs)
                elif X.is_op(opns.DENSE):
                    A, W = X.args
                    # (A * W) * gamma + bias
                    # A * (W * gamma) + bias
                    W_data = self.get_as_numpy(W) * Gamma.reshape(K, 1)
                    W_sym = self.from_np_data(W, W_data, W.dtype)
                    out = op.nn_dense(A, W_sym, **X.attrs)
                else:
                    reshp = [s if i == sym.axis else 1 \
                            for i, s in enumerate(X.shape)]
                    W = self.from_np_data(X, Gamma.reshape(reshp), X.dtype)
                    out = opclass.mul(X, W)

                bias = bias.reshape([s if i == sym.axis else 1 \
                        for i, s in enumerate(out.shape)])
                B = out.like(sym)
                B = self.from_np_data(B, bias, dtype=B.dtype)
                return opclass.add(out, B).like(sym)

            return sym
        return custom_run


class FuseDividePass(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            if sym.op_name == opns.DIV:
                argA = sym.args[0]
                argB = sym.args[1]
                assert self.is_param(argB), f'NotParam: {argB}'
                argB = self.from_np_data(sym, 1. / self.get_as_numpy(argB), dtype=argB.dtype)
                out = opclass.mul(argA, argB)
                return out.like(sym)
            return sym
        return custom_run


class FuseLeakyReLU(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            if sym.op_name == opns.LEAKY_RELU:
                alpha = self.from_const_data(sym, sym.alpha, dtype=float)
                X = sym.args[0]
                out = opclass.relu(opclass.negative(X))
                out = opclass.mul(alpha, out)
                return opclass.sub(opclass.relu(X), out)
            return sym
        return custom_run

class FuseAdaptiveAvgPool2D(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            if sym.op_name == opns.ADAPTIVE_AVG_POOL2D:
                X = sym.args[0]
                assert sym.layout == "NCHW"
                inp_shap = X.shape[2:]
                out_size = sym.output_size or inp_shap
                if not isinstance(out_size, (list, tuple)):
                    out_size = (out_size, out_size)
                sym.output_size = out_size

                assert len(X.shape) == 4
                if all([s == 1 for s in sym.output_size]):
                    scale = np.array(1 / np.prod(X.shape[-2:]))
                    out = opclass.Sum(X, dim=list(range(4))[-2:], keepdims=True)
                    scale = self.from_np_data(sym, scale.astype(X.dtype))
                    return opclass.mul(out, scale).like(self)
                elif out_size[0] > inp_shap[0] or out_size[1] > inp_shap[1]:
                    assert all([s == 1 for s in inp_shap])
                    # TODO: fix opclass repeat
                    out = opclass.repeat(X, repeats=out_size[0], axis=-2)
                    out = opclass.repeat(out, repeats=out_size[1], axis=-1)
                    return out.like(self)

                # calculate the attributes refers to:
                # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work
                strides = [i // o for i, o in zip(inp_shap, out_size)]
                kernel = [i-(o-1)*s for i, o, s in zip(inp_shap, out_size, strides)]
                attrs = {
                    "kernel_size": kernel,
                    "strides": strides,
                    "padding": (0, 0),
                    "dilation": (1, 1),
                    "data_layout": sym.layout,
                    "groups": X.shape[1],
                    "channels": X.shape[1],
                }
                W_shape = (X.shape[1], 1, *kernel)
                W = self.from_np_data(X, np.full(W_shape, 1 / product(kernel)), dtype=X.dtype)
                out = opclass.Conv2D(X, W, **attrs)
                return out.like(sym)
            return sym
        return custom_run
    

class FuseAvgPool2D(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            return sym
        return custom_run

class Spliter(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            return sym
        return custom_run

class Merger(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            return sym
        return custom_run

class Calibrator(InferPass):
    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:
            return sym
        return custom_run
