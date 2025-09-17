from __future__ import annotations
import typing

from functools import wraps
from dataclasses import dataclass, fields

import mrt
from mrt.common import config
from mrt.common.utils import *
from mrt.common.types import *

from . import opns, opclass
from . import symbol as _symbol


# mrt op visits
class SimplePass:
    symbol: _symbol.Symbol

    def __init__(self, symbol: _symbol.Symbol):
        self.symbol = symbol

    def visit(self, custom_func: typing.Callable[[Symbol], typing.Optional[Symbol]] = None) -> _symbol.Symbol:
        env: typing.Dict[str, _symbol.Symbol] = {}
        for sym in _symbol.sym2list(self.symbol):
            assert sym.name not in env, f'{sym.name} NotIn env!'

            # Updating args as passed symbol in env_dict
            sym = sym.copy(args = [env[arg_sym.name] for arg_sym in sym.args])
            assert isinstance(sym, _symbol.Symbol), sym
            out = custom_func(sym) if custom_func else getattr(self, f"visit_{opns.Opname2Funcname(sym.op_name)}")(sym)
            out = out or sym
            assert isinstance(out, _symbol.Symbol), out
            env[sym.name] = out
        return env[self.symbol.name]

    def _default_visit_op(self, op: _symbol.Symbol) -> _symbol.Symbol:
        return op


# mrt op visits with params, variables
class InferPass(SimplePass):
    params: ParametersT

    def is_param(self, symbol: _symbol.Symbol) -> bool:
        return symbol.op_name == opns.VAR and symbol.name in self.params

    def get_param(self, symbol: _symbol.Symbol) -> OpNumpyT:
        assert self.is_param(symbol)
        return self.params[symbol.name] if self.is_param(symbol) else []

    def __init__(self, symbol: _symbol.Symbol, params: ParametersT):
        self.symbol = symbol
        self.params = params


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
    def visit_TupleGetItem(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.TUPLE_GET_ITEM:
            return sym
            sym_ : opclass.TupleGetItem = sym
            assert sym_.index == 0
            return sym_.args[0]
        return sym


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


class FuseSoftmaxPass(SimplePass):
    def visit_nn_softmax(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.SOFTMAX:
            return self.args[0]
        return sym

    def visit_nn_log_softmax(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.LOG_SOFTMAX:
            return self.args[0]
        return sym


class FuseMeanPass(SimplePass):
    def visit_mean(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.MEAN:
            return sym
        return sym


class FuseDividePass(InferPass):
    def visit_divide(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.DIV:
            argA = sym.args[0]
            argB = sym.args[1]
            assert self.is_param(argB), f'NotParam: {argB}'
            # TODO: fixit
            #argB = argB.from_np_data(1. / argB.numpy())
            return opclass.MRT_OP_MAP[opns.MUL](argA, argB)
        return sym

