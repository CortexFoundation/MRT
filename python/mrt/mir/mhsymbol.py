import typing

from mrt.common.utils import *
from mrt.common.types import *

from . import opns, opclass, optype
from . import symbol

#from mrt.mir.mhsymbol import MultiHeadSymbol, Graph
class MultiHeadSymbol(dict):
    """ { "main": F(X) } """
    origin: typing.Optional[symbol.Symbol] = None

    @classmethod
    def from_symbol(cls, symbol: symbol.Symbol, name: str = "main"):
        return MultiHeadSymbol({ name: symbol })

    def as_tuple(self) -> typing.Tuple[typing.List[str], symbol.Symbol]:
        from . import op
        #  args = list(self.values())
        #  sym_type = type(args[0]) if args else Symbol
        mhs = self.origin or optype.infer_single(opclass.MRT_OP_MAP[opns.TUPLE](*list(self.values())))
        return list(self.keys()), mhs

    @classmethod
    def from_tuple(cls, tuple_names, symbol):
        assert symbol.is_op(opns.TUPLE), symbol
        mhs = cls(zip(tuple_names, symbol.args))
        mhs.origin = symbol
        return mhs

Graph = typing.Union[symbol.Symbol, MultiHeadSymbol]
""" Notice that Symbol and MultiHeadSymbol can both
        be regarded as a model Graph.
"""

