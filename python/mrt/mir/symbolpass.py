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
class SymbolPass:
    symbol: _symbol.Symbol
    params: ParametersT
    
    def __init__(self, symbol: _symbol.Symbol, params: ParametersT):
        self.symbol = symbol
        self.params = params

    def is_param(self, symbol: _symbol.Symbol) -> bool:
        return symbol.op_name == opns.VAR and symbol.name in self.params

    def visit(self) -> _symbol.Symbol:
        env: typing.Dict[str, _symbol.Symbol] = {}
        for sym in _symbol.sym2list(self.symbol):
            assert sym.name not in env, f'{sym.name} NotIn env!'

            # Updating args as passed symbol in env_dict
            sym = sym.copy(args = [env[arg_sym.name] for arg_sym in sym.args])
            assert isinstance(sym, _symbol.Symbol), sym

            if sym.op_name == opns.DROP_OUT:
                #print('ddrroopped_out', getattr(self, f"visit_{opns.Opname2Funcname(sym.op_name)}")(sym) or sym)
                pass
            out = getattr(self, f"visit_{opns.Opname2Funcname(sym.op_name)}")(sym) or sym
            assert isinstance(out, _symbol.Symbol), out
            env[sym.name] = out
        return env[self.symbol.name]

    def _default_visit_op(self, op: _symbol.Symbol) -> _symbol.Symbol:
        return op


# register mrt op default_visit
for op_name in opns.MRT_OP_SET:
    funcSuffix = opns.Opname2Funcname(op_name)
    setattr(SymbolPass, f"visit_{funcSuffix}", SymbolPass._default_visit_op)
    #print(f"visit_, {op_name} => {funcSuffix}", getattr(SymbolPass, f"visit_{funcSuffix}"))


# mrt symbol pass
class FuseDropoutPass(SymbolPass):
    def visit_nn_dropout(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        # make sure op fit again
        if sym.op_name == opns.DROP_OUT:
            return sym.args[0]
        return sym


class FuseDividePass(SymbolPass):
    def visit_divide(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.DIVIDE:
            argA = self.args[0]
            argB = self.args[1]
            assert self.is_param(argB), f'NotParam: {argB}'
            # TODO: fixit
            #argB = argB.from_np_data(1. / argB.numpy())
            return opclass.Multiply(sym.name, {'args':[argA, argB]})
        return sym


class FuseTupleGetItemPass(SymbolPass):
    def visit_TupleGetItem(self, sym: _symbol.Symbol) -> _symbol.Symbol:
        if sym.op_name == opns.TUPLE_GET_ITEM:
            sym_ : opclass.TupleGetItem = sym
            assert sym_.index == 0
            return sym_.args[0]
        return sym

