from __future__ import annotations

import typing
from dataclasses import dataclass, field

from . import utils

T = typing.TypeVar("T")

@dataclass
class _BaseConfig:
    _last_scope: _BaseConfig | typing.Type[_BaseConfig] | None = None
    __GLOBAL__: typing.ClassVar[_BaseConfig | typing.Type[_BaseConfig] | None] = None

    def __post_init__(self):
        self._last_scope = self.__GLOBAL__

    @classmethod
    def G(cls: typing.Type[T]) -> T:
        return cls.__GLOBAL__ or cls() # type: ignore

    @classmethod
    def _set_scope(cls: typing.Type[T], ins) -> T:
        assert isinstance(ins, cls)
        cls.__GLOBAL__ = ins # type: ignore
        return ins

    def __enter__(self):
        return self._set_scope(self)

    def __exit__(self, *_) -> bool:
        self._set_scope(self._last_scope)
        # return false to indicate throw exception
        return False

    def register_global(self):
        return self._set_scope(self)

    def mutate(self, **new_attrs):
        attrs = utils.dataclass_to_dict(self)
        attrs.update(new_attrs)
        return type(self)(**attrs)


@dataclass
class PassConfig(_BaseConfig):
    name        : str   = ""
    inherit     : bool  = False
    """ whether to inherit config in iterate pass. """
    log_before  : bool  = False
    log_after   : bool  = False

SYMBOL_ALL_NEAR = "__symbol_all_near_flag__"
""" this flag always return true for Symbol.is_near."""

@dataclass
class MRTConfig(_BaseConfig):
    max_const_size: int = 1
    """ use relax.const if shape product less/equal than this val.

        Invalid const size may leads to tvm compile failed:
            Memory verification failed
    """

@dataclass
class LogConfig(_BaseConfig):
    log_vot_cbs: typing.List[str] = field(default_factory=list)
    """ log visit or transform func callbacks. """

    log_type_infer: typing.List[str] = field(default_factory=list)
    """ log symbol that is near these names. """

    log_cast_relax: bool = False
    """ log in expr2symbol, since it use tvm topo sort. """

    fill: str = " "
    name_align: str = ">"
    name_width: int = 10

    seg_join: str = " | "
    seg_align: str = "<"
    seg_width: int = 15


def log_str(name: str, *segs) -> str:
    C = LogConfig.G()
    segs = [str(s) for s in segs]
    msg = [f"{s:{C.fill}{C.seg_align}{C.seg_width}}" for s in segs]
    msg = C.seg_join.join(msg)
    msg = f"[{name:{C.fill}{C.name_align}{C.name_width}}] {msg}"
    return msg

def log(*args, **kwargs):
    print(log_str(*args, **kwargs))

