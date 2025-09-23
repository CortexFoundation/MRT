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

    def from_np_data(self, data: np.ndarray, dtype, prefix=None) -> _symbol.Symbol:
        name = N.n(prefix=prefix)
        # some data is np.float/int type, use np.array to wrap it.
        data = np.array(data)
        self.params[name] = data.astype(dtype)
        return opclass.var(name, shape=data.shape, dtype=dtype)#.like(self)

    def from_const_data(self, data: typing.Union[int, float], dtype) -> _symbol.Symbol:
        return self.from_np_data(data, dtype)


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
    def visit_mean(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.MEAN:
            X = sym.args[0]
            out = opclass.Sum(X, **sym.attrs)
            scale = self.from_np_data(np.array(
                1. * product(out.shape) / product(X.shape)), dtype=out.dtype)
            out = opclass.mul(out, scale)
            return out
        return sym


class FuseConstantPass(InferPass):
    threshold: typing.ClassVar[float] = 1e-5

    def np_is_zero(self, data) -> float:
        return np.abs(data).max() < self.threshold


    def get_run(self) -> _symbol._TransformerParamT:
        def custom_run(sym: _symbol.Symbol, params: typing.Optional[ParametersT] = None) -> _symbol.Symbol:#: _symbol._TransformerParamT
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
                    return None
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
    def visit_nn_batch_norm(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.BATCH_NORM:
            X, Gamma, Beta, Mean, Var = sym.args
            Gamma = self.get_param(Gamma)
            Beta = self.get_param(Beta)
            Mean = self.get_param(Mean)
            Var = self.get_param(Var)
            return sym
        return sym


class FuseDividePass(InferPass):
    def visit_divide(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.DIV:
            argA = sym.args[0]
            argB = sym.args[1]
            assert self.is_param(argB), f'NotParam: {argB}'
            argB = self.from_np_data(1. / self.get_as_numpy(argB), dtype=argB.dtype)
            return opclass.MRT_OP_MAP[opns.MUL](argA, argB)
        return sym

