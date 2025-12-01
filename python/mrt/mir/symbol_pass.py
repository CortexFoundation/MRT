from __future__ import annotations

import typing
from functools import wraps
from dataclasses import field

import numpy as np

from mrt.mir.symbol import *
from mrt.mir.mhsymbol import Graph

from mrt.mir import op, opns, opclass
from mrt.mir.attrs import _BaseAttrs, parse_attrs

from mrt.common.utils import N


class SymbolBridge: # SymbolManipulator / Pass
    graph: Symbol

    def __init__(self, symbol: Symbol):
        self.graph = symbol

    @classmethod
    def base(cls, symbol: Symbol):
        return cls(symbol)

    def __repr__(self, **attrs):
        return self.graph.__repr__(**attrs)

    def from_symbol(self, sym: Symbol) -> typing.Self:
        return type(self)(sym)

    @property
    def parsed(self)-> _BaseAttrs:
        return parse_attrs(self.graph.op_name, self.graph.attrs)
        return self.graph.attrs

    """Member Symbol Start
    """
    def is_op(self, *op_names) -> bool:
        """ Check current symbol is in the op name list. """
        assert len(op_names) > 0
        return self.graph.op_name in op_names
    def is_near(self, *names, check_args: bool = True) -> bool:
        return self.graph.is_near(*names, check_args)
    def to_dict(self) -> dict:
        return self.graph.to_dict()
    @classmethod
    def from_dict(cls, d: dict, **kwargs) -> SymbolParameters:
        return cls(Symbol.from_dict(d, **kwargs), {})
    @property
    def args(self) -> list:
        return self.graph.args
    @property
    def op_name(self) -> str:
        return self.graph.op_name
    @property
    def name(self) -> str:
        return self.graph.name
    @property
    def shape(self) -> typing.Optional[ShapeT]:
        return self.graph.shape
    @property
    def dtype(self) -> str:
        return self.graph.dtype
    @property
    def attrs(self) -> dict:
        return self.graph.attrs
    @property
    def extra_attrs(self) -> dict:
        return self.graph.extra_attrs
    def set_extra_attrs(self, **kwargs):
        return self.graph.extra_attrs.update(kwargs)
    """Member Symbol End
    """

class SymbolParameters(SymbolBridge):
    graph: Symbol
    params: ParametersT = field(repr=False)
    """ Parameters should not be changed in transformer,
            use copy mode instead to avoid possible errors.

        deep copy params in trace `checkpoint_run` api.
    """

    def __init__(self, symbol: Symbol, params: ParametersT):
        self.graph = symbol
        self.params = params

    @classmethod
    def base(cls, symbol: Symbol, params: ParametersT):
        return cls(symbol, params)

    def __repr__(self, **attrs):
        if self.is_param():
            attrs["absmax"] = np.abs(self.numpy()).max(initial=0)
        return super().__repr__(**attrs)

    @property
    def parsed(self)-> _BaseAttrs:
        return parse_attrs(self.graph.op_name, self.graph.attrs)
        attrs = self.graph.attrs
        return attrs


    def numpy(self) -> OpNumpyT:
        assert self.is_param(), f"{self.graph.name} is not parameter."
        data = self.params[self.graph.name]
        assert isinstance(data, (tuple, list, np.ndarray)), \
                f"param:{self.graph.name} not OpNumpyT, get {type(data)}"
        return data

    def as_parameter(self, data: OpNumpyT) -> Symbol:
        def _f(data, dtype):
            if isinstance(data, list):
                assert len(data) == len(dtype)
                return [_f(d, t) for d, t in zip(data, dtype)]
            assert isinstance(data, np.ndarray), type(data)
            return data.astype(dtype)

        self.params[self.graph.name] = _f(data, self.graph.dtype)
        return op.as_variable(self.graph)

    def from_const_data(self, data: typing.Union[int, float]) -> Symbol:
        return self.from_np_data(data)

    def from_symbol(self, sym: Symbol) -> typing.Type[SymbolParameters]:
        return type(self)(sym, self.params)

    def from_np_data(self, data: np.ndarray | typing.Union[int, float], prefix="%") -> Symbol:
        """ out = Return Symbol
            out = op.add(out, B)
            self: SymbolParameter
            self.graph: Symbol
            self.from_symbol(out).from_np_data()

            out = Return Symbol
            out.from_np_data()

            op.add(out.graph, B)

            graph: Symbol
        """
        name = N.n(prefix=prefix)
        # some data is np.float/int type, use np.array to wrap it.
        data = np.array(data)
        self.params[name] = data.astype(self.graph.dtype)
        ## return type(self). # Mark!
        return opclass.var(name, data.shape, self.graph.dtype).like(self.graph)

    def is_input(self) -> bool:
        return op.is_input(self.graph, self.params)
    def is_param(self) -> bool:
        return op.is_param(self.graph, self.params)
    def is_variable(self) -> bool:
        return op.is_variable(self.graph, self.params)
    def is_operator(self) -> bool:
        return op.is_operator(self.graph, self.params)

SymTransformerT = typing.Callable[[Graph], Graph]
""" Symbol-Transformer Callback Function Type,
        inherited from SymbolParameters.
"""

class SymbolTransformer(SymbolParameters):
    """ Symbol Transformer(Manipulator) """

    RUN_ONCE: typing.ClassVar[bool] = False

    # inherit SymbolParameters __init__
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_transformer(cls, name: typing.Optional[str] = None):
        name = name or cls.__name__
        def _func(graph: Symbol, params: ParametersT, **kwargs):
            def _run(sym: Symbol):
                # use current cls to apply transform, this
                #   may loss some information from origin
                #   symbol, so record as `origin` in call.
                out = cls.base(sym, params) # Type as SymbolTransformer
                out = out(origin=sym, **kwargs) or sym # Type as Symbol
                assert isinstance(out, Symbol), (
                        "transform output type should be {},"
                        " but get {}"
                        ).format(cls, type(out))
                return out
            _run.__name__ = name
            with N(name):
                return _run(graph) if cls.RUN_ONCE \
                        else transform(graph, _run)
        _func.__name__ = name
        return _func

    # @classmethod
    # def apply(cls, *args, **kw):
    #     """ Static apply function to generator transformer pass.

    #     All the parameters are used to invoke `call` method.
    #     """
    #     def _tfm(sym: Symbol, params: ParametersT):
    #         ins = cls.base(sym, params=params)
    #         out = ins(*args, **kw) or ins
    #         assert isinstance(out, cls), (
    #             "expected {}, but get {}"
    #                 ).format(cls, type(out))
    #         return out

    #     _tfm.__name__ = cls.__name__
    #     return _tfm

    def __call__(self, *args, **kw) -> typing.Optional[SymbolTransformer]:
        """
            Parameters:
            origin: original symbol passed from last transformer.
        """
        raise NotImplementedError()

class RunOnce(SymbolTransformer):
    RUN_ONCE: typing.ClassVar[bool] = True

    def __init__(self, *args): # symbol: Symbol, params: ParametersT):#, parsed: _BaseAttrs=None):
        super().__init__(*args)

