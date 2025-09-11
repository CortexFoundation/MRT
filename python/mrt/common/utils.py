import os
import math
import typing
import threading
from os import path

from .types import *

ROOT = path.abspath(path.join(__file__, "../../../"))
PY_ROOT = path.join(ROOT, "python")

MRT_MODEL_ROOT = path.expanduser("~/mrt_model")
if not path.exists(MRT_MODEL_ROOT):
    os.makedirs(MRT_MODEL_ROOT)

MRT_DATASET_ROOT = path.expanduser("~/.mxnet/datasets")
if not path.exists(MRT_DATASET_ROOT):
    os.makedirs(MRT_DATASET_ROOT)

SelfScope = typing.TypeVar("SelfScope", bound="Scope")

class Scope:
    """
        1. Scope may have many instance
        2. Scope can enter/exit recursively
        3. Scope has Global instance by default.
    """
    __CURR_GLOBAL_INSTANCE__: SelfScope | None = None

    def __init__(self):
        """ Scope will auto record current global scope. """
        self.last_scope = self.__CURR_GLOBAL_INSTANCE__

    def __enter__(self) -> SelfScope:
        self.register_global(self)
        return self

    def __exit__(self, *args):
        """ Trace back last scope """
        self.register_global(self.last_scope)

    @classmethod
    def register_global(cls, ins: SelfScope) -> SelfScope:
        """ register global scope """
        cls.__CURR_GLOBAL_INSTANCE__ = ins
        return ins

    @classmethod
    def G(cls) -> SelfScope:
        """ Get Current Global Instance. """
        sc = cls.__CURR_GLOBAL_INSTANCE__ or cls()
        sc.register_global(sc)
        return sc

class N(Scope):
    def __init__(self, name=""):
        super().__init__()
        self.counter = 0
        self.scope_name = name
        self.lock = threading.Lock()

    def _alloc_name(self, prefix, suffix) -> str:
        with self.lock:
            index = self.counter
            self.counter += 1
        name = "{}{}{}".format(prefix, index, suffix)
        if self.scope_name:
            name = "{}.{}".format(self.scope_name, name)
        return name

    @staticmethod
    def n(prefix="%", suffix="") -> str:
        ins: N = N.G()
        return ins._alloc_name(prefix, suffix)

def extend_fname(prefix, with_ext=False):
    """ Get the precision of the data.

        Parameters
        __________
        prefix : str
            The model path prefix.
        with_ext : bool
            Whether to include ext_file path in return value.

        Returns
        _______
        ret : tuple
            The symbol path, params path; and with_ext is True, also return ext file path.
    """
    ret = ["%s.json"%prefix, "%s.params"%prefix]
    if with_ext:
        ret.append("%s.ext"%prefix)
    return tuple(ret)

from dataclasses import dataclass, fields, Field, is_dataclass
def dataclass_to_dict(dc: dataclass, check_repr=False) -> dict:
    def _check(f: Field):
        checked = True
        if check_repr:
            checked = checked and f.repr
        return checked
    return {f.name: getattr(dc, f.name) \
            for f in fields(dc) if _check(f)}
    # return dict((f.name, getattr(dc, f.name)) \
    #         for f in fields(dc))

def load_dc_attrs_from_env(dc: dataclass) -> dict:
    """ load dataclass config from environment variables.
        1. env key is uppercase of field name.
        2. multi-value is splited with comma for list types.
    """
    type_hints = typing.get_type_hints(dc)
    new_attrs = {}
    for f in fields(dc):
        env_key = f.name.upper()
        if env_key not in os.environ:
            continue

        env_val = os.environ[env_key]
        field_type = type_hints[f.name]
        origin = typing.get_origin(field_type) or field_type
        if origin is list:
            val = env_val.split(',')
        elif origin is bool:
            val = env_val.lower() in ['true', '1', 't', 'y', 'yes']
        elif origin is int:
            val = int(env_val)
        elif origin is float:
            val = float(env_val)
        else: # str or other types
            val = env_val
        new_attrs[f.name] = val
    return new_attrs

def product(arr_like):
    """ calculate production for input array.

        if array is empty, return 1.
    """
    total = 1
    for s in arr_like:
        total *= s
    return total

def number_to_bits(number: float) -> int:
    """ Return the integer bits to represent number.
        precision bit: 1
        number bits:
            [ 0-0 ] => 0, skip
            [ 1-1 ] => 1, ceil(log2(i+1)) = 1
            [ 2-3 ] => 2, ceil(log2(i+1)) = 2
            [ 4-7 ] => 3, ceil(log2(i+1)) = 3
            ...

        return 1 + ceil(log2(number + 1))

        note: consider the abs round int for number.
    """
    number = math.fabs(number)
    number = math.floor(number + 0.5)
    return 1 + math.ceil(math.log2(number + 1))

def bits_to_number(bit: int) -> float:
    assert bit > 0
    return float((2 ** (bit - 1)) - 1)

def count_to_bits(count: int):
    """
    # get_bit_cnt (mrt) should be consistent with
    # GetReduceSumBit (cvm-runtime)

    """
    prec = 0
    while count != 0:
        prec += 1
        count >>= 1
    return prec

def get_class_name(o):
    klass = o if isinstance(o, type) else o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__

