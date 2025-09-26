import typing

from mrt.common.types import *
from mrt.common import config

#  from . import config
from . import op, opns
from .symbol import Symbol, transform

InferTypeT = typing.Callable[[Symbol], Symbol]
_INFER_TYPE_REG: typing.Dict[str, InferTypeT] = {}
_DEFAULT_TYPE_INFER: InferTypeT = lambda x: x

def register_type_infer(
        *op_names,
        rule: typing.Optional[InferTypeT] = None):
    def _set_rule(func: InferTypeT):
        for op in op_names:
            _INFER_TYPE_REG[op] = func
        return func

    if rule is not None:
        return _set_rule(rule)
    return _set_rule

def infer_single(symbol: Symbol) -> Symbol:
    C: config.LogConfig = config.LogConfig.G()

    out = op.retrieve_operator(symbol)

    from mrt.frontend import api
    # use frontend type_infer api as fallback.
    _infer = _INFER_TYPE_REG.get(out.op_name, api.type_infer)
    assert _infer is not None

    if symbol.is_near(*C.log_type_infer):
        config.log(
                out.name,
                f"{out.name}@{out.op_name}",
                f"use tinfer func: {_infer.__name__}")

    out: Symbol = _infer(out) or out
    assert out.shape is not None, out
    assert out.dtype is not None, out
    type_info = { "shape": out.shape, "dtype": out.dtype }
    return symbol.copy(extra_attrs={
        **symbol.extra_attrs, **type_info })

def infer(symbol: Symbol) -> Symbol:
    return transform(symbol, infer_single)

@register_type_infer(opns.ARGWHERE)
def _argwhere(symbol: Symbol) -> Symbol:
    X: Symbol = symbol.args[0]
    symbol.shape = X.shape
    symbol.dtype = "int32"

@register_type_infer(opns.TUPLE)
def _tuple(symbol: Symbol) -> Symbol:
    symbol.shape = [a.shape for a in symbol.args]
    symbol.dtype = [a.dtype for a in symbol.args]

@register_type_infer(opns.TUPLE_GET_ITEM)
def _tuple_get_item(symbol: Symbol) -> Symbol:
    X: Symbol = symbol.args[0]
    index = symbol.attrs["index"]
    symbol.shape = X.shape[index]
    symbol.dtype = X.dtype[index]

@register_type_infer(opns.REQUANT, opns.PCLIP, opns.RS_PCLIP)
def _type_like_first(symbol: Symbol) -> Symbol:
    X: Symbol = symbol.args[0]
    symbol.shape = X.shape
    symbol.dtype = X.dtype
