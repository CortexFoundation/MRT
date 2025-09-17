import typing
import numpy as np
from dataclasses import dataclass, fields

from mrt.common.utils import N
from . import opns
from . import symbol
from .symbol import SelfSymbol

#SelfSymbol = typing.TypeVar("SelfSymbol", bound="Symbol")
MRT_OP_MAP: typing.Dict[str, SelfSymbol] = {}

def _register_op_map(op_name: str):
    def _wrapper(clss: SelfSymbol = None) -> SelfSymbol:
        if len(op_name) > 0 and clss != None:
            if op_name not in MRT_OP_MAP:
                MRT_OP_MAP[op_name] = clss
            else:
                print(f'Warning: "{op_name}" Alreary Registered In MRT_OP_MAP, IsBeing Overrided!')
                MRT_OP_MAP[op_name] = clss
        return clss
    return _wrapper


@dataclass(init=False)
class Variable(symbol.Symbol):
    op_name = opns.VAR

    def __init__(self, name=None, op_name=None, shape:typing.Tuple = (), dtype=None, extra_attrs=None):
        op_name = op_name or opns.VAR
        assert op_name == opns.VAR
        super().__init__(name=name or N.n(), op_name=op_name, args=[], attrs={}, extra_attrs=extra_attrs or {})
        self.shape = shape # will also update extra_attrs
        self.dtype = dtype # will also update extra_attrs

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        data = cls.default_dict()
        data.update(d)
        data.update(kwargs)
        data = cls.update_dict(data)
        basedata = {k: data[k] for k in data if k in ['name', 'op_name', 'extra_attrs']}
        attrsdata = {k: data['extra_attrs'][k] for k in data['extra_attrs'] if k in ['shape', 'dtype']}
        try:
            out = cls(**attrsdata, **basedata)
        except Exception as e:
            raise e
        return out

 
@dataclass(init=False)
class Conv2D(symbol.Symbol):
    op_name = opns.CONV2D

    @property
    def strides(self) -> typing.Tuple[int, int]:
        default_val = (1,1)
        return self.attrs['strides'] if 'strides' in self.attrs else default_val

    @property
    def padding(self) -> typing.Tuple[int, int, int, int]:
        default_val = (0,0,0,0)
        return self.attrs['padding'] if 'padding' in self.attrs else default_val

    @property
    def groups(self) -> int:
        default_val = 1
        return self.attrs['groups'] if 'groups' in self.attrs else default_val

    @property
    def dilation(self) -> typing.Tuple[int, int]:
        default_val = (1,1)
        return self.attrs['dilation'] if 'dilation' in self.attrs else default_val

    @property
    def kernel_size(self) -> typing.Tuple[int, int]:
        default_val = (3,3)
        return self.attrs['kernel_size'] if 'kernel_size' in self.attrs else default_val


    # Follows (*args, name, **attrs)
    def __init__(self, X, W, name=None, op_name=None, strides=(1,1), padding=(0,0,0,0), groups=1, dilation=(1,1), kernel_size=(3,3), extra_attrs=None):
        op_name = op_name or opns.CONV2D
        assert op_name == opns.CONV2D
        super().__init__(name=name or N.n(), op_name=op_name, args=[X,W], attrs={'strides':strides, 'padding':padding, 'groups':groups, 'dilation':dilation, 'kernel_size':kernel_size}, extra_attrs=extra_attrs or {})


    # Copy from other instance of same opclass, must have specific attrs (or with default value)
    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        data = cls.default_dict()
        data.update(d)
        data.update(kwargs)
        data = cls.update_dict(data)
        basedata = {k: data[k] for k in data if k in ['name', 'op_name', 'extra_attrs']}
        attrsdata = {k: data['attrs'][k] for k in data['attrs'] if k in ['strides', 'padding', 'groups', 'dilation', 'kernel_size']}
        try:
            out = cls(data['args'][0], data['args'][1], **attrsdata, **basedata)
        except Exception as e:
            raise e
        return out

@dataclass(init=False)
class Dropout(symbol.Symbol):
    op_name = opns.DROP_OUT

    @property
    def rate(self) -> float:
        default_val = 0.0
        return self.attrs['rate'] if 'rate' in self.attrs else default_val 
    
    def __init__(self, X, name=None, op_name=None, rate:float = 0, extra_attrs=None):
        op_name = op_name or opns.DROP_OUT
        assert op_name == opns.DROP_OUT
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'rate': rate}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        data = cls.default_dict()
        data.update(d)
        data.update(kwargs)
        data = cls.update_dict(data)
        basedata = {k: data[k] for k in data if k in ['name', 'op_name', 'extra_attrs']}
        attrsdata = {'rate': data['attrs']['rate']}
        try:
            out = cls(data['args'][0], **attrsdata, **basedata)
        except Exception as e:
            raise e
        return out

@dataclass(init=False)
class Clip(symbol.Symbol):
    op_name = opns.CLIP
    
    @property
    def min(self) -> float:
        default_val = np.nan
        return self.attrs['min'] if 'min' in self.attrs else default_val

    @property
    def max(self) -> float:
        default_val = np.nan
        return self.attrs['max'] if 'max' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, min_:float = np.nan, max_:float = np.nan, extra_attrs=None):
        op_name = op_name or opns.CLIP
        assert op_name == opns.CLIP
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'min': min_, 'max': max_}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        data = cls.default_dict()
        data.update(d)
        data.update(kwargs)
        data = cls.update_dict(data)
        basedata = {k: data[k] for k in data if k in ['name', 'op_name', 'extra_attrs']}
        attrsdata = {'min': data['attrs']['min'], 'max': data['attrs']['max']}
        try:
            out = cls(data['args'][0], **attrsdata, **basedata)
        except Exception as e:
            raise e
        return out


@dataclass(init=False)
class BatchNorm(symbol.Symbol):
    op_name = opns.BATCH_NORM

    @property
    def axis(self) -> int:
        default_val = 1
        return self.attrs['axis'] if 'axis' in self.attrs else default_val

    @property
    def epsilon(self) -> float:
        default_val = 1e-5
        return self.attrs['epsilon'] if 'epsilon' in self.attrs else default_val

    @property
    def center(self) -> bool:
        default_val = True
        return self.attrs['center'] if 'center' in self.attrs else default_val

    @property
    def scale(self) -> bool:
        default_val = True
        return self.attrs['scale'] if 'scale' in self.attrs else default_val

    def __init__(self, X, Gamma, Beta, Mean, Var, name=None, op_name=None, axis:int = 1, epsilon:float = 1e-5, center:bool = True, scale:bool = True, extra_attrs=None):
        op_name = op_name or opns.BATCH_NORM
        assert op_name == opns.BATCH_NORM
        super().__init__(name=name or N.n(), op_name=op_name, args=[X, Gamma, Beta, Mean, Var], attrs={'axis': axis, 'epsilon': epsilon, 'center': center, 'scale': scale}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        data = cls.default_dict()
        data.update(d)
        data.update(kwargs)
        data = cls.update_dict(data)
        basedata = {k: data[k] for k in data if k in ['name', 'op_name', 'extra_attrs']}
        attrsdata = {k: data['attrs'][k] for k in data['attrs'] if k in ['axis', 'epsilon', 'center', 'scale']}
        try:
            out = cls(*data['args'], **attrsdata, **basedata)
        except Exception as e:
            raise e
        return out

 
@dataclass(init=False)
class TupleGetItem(symbol.Symbol):
    op_name = opns.TUPLE_GET_ITEM
    
    @property
    def index(self) -> float:
        default_val = 0
        return self.attrs['index'] if 'index' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, index:int = 0, extra_attrs=None):
        op_name = op_name or opns.TUPLE_GET_ITEM
        assert op_name == opns.TUPLE_GET_ITEM
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'index': index}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        data = cls.default_dict()
        data.update(d)
        data.update(kwargs)
        data = cls.update_dict(data)
        basedata = {k: data[k] for k in data if k in ['name', 'op_name', 'extra_attrs']}
        attrsdata = {'index': data['attrs']['index']}
        try:
            out = cls(data['args'][0], **attrsdata, **basedata)
        except Exception as e:
            raise e
        return out


_register_op_map(opns.VAR)(Variable)
_register_op_map(opns.CONV2D)(Conv2D)
_register_op_map(opns.DROP_OUT)(Dropout)
_register_op_map(opns.CLIP)(Clip)
_register_op_map(opns.BATCH_NORM)(BatchNorm)
_register_op_map(opns.TUPLE_GET_ITEM)(TupleGetItem)

# Add default register Class for MRT OP Not Implemented!
_register_op_map(opns.MUL)(symbol.Symbol)
_register_op_map(opns.DENSE)(symbol.Symbol)
_register_op_map(opns.RELU)(symbol.Symbol)
_register_op_map(opns.HARDTANH)(symbol.Symbol)
_register_op_map(opns.SILU)(symbol.Symbol)
_register_op_map(opns.LEAKY_RELU)(symbol.Symbol)
_register_op_map(opns.ADAPTIVE_AVG_POOL2D)(symbol.Symbol)
_register_op_map(opns.AVG_POOL2D)(symbol.Symbol)
_register_op_map(opns.MAX_POOL2D)(symbol.Symbol)
_register_op_map(opns.SOFTMAX)(symbol.Symbol)
_register_op_map(opns.LOG_SOFTMAX)(symbol.Symbol)
_register_op_map(opns.EXP)(symbol.Symbol)
_register_op_map(opns.SIGMOID)(symbol.Symbol)
_register_op_map(opns.SUM)(symbol.Symbol)
_register_op_map(opns.MEAN)(symbol.Symbol)
_register_op_map(opns.MAX_AXIS)(symbol.Symbol)
_register_op_map(opns.MAXIMUM)(symbol.Symbol)
_register_op_map(opns.MINIMUM)(symbol.Symbol)
_register_op_map(opns.TUPLE)(symbol.Symbol)
_register_op_map(opns.REPEAT)(symbol.Symbol)
_register_op_map(opns.SQUEEZE)(symbol.Symbol)
_register_op_map(opns.FLATTEN)(symbol.Symbol)
_register_op_map(opns.BATCH_FLATTEN)(symbol.Symbol)
_register_op_map(opns.RESHAPE)(symbol.Symbol)
_register_op_map(opns.CONCAT)(symbol.Symbol)
_register_op_map(opns.SPLIT)(symbol.Symbol)
_register_op_map(opns.TRANSPOSE)(symbol.Symbol)
_register_op_map(opns.BROADCAST_TO)(symbol.Symbol)
_register_op_map(opns.EXPAND_DIMS)(symbol.Symbol)
_register_op_map(opns.TILE)(symbol.Symbol)
_register_op_map(opns.WHERE)(symbol.Symbol)
_register_op_map(opns.GREATER)(symbol.Symbol)
_register_op_map(opns.STRIDED_SLICE)(symbol.Symbol)
_register_op_map(opns.SLICE_LIKE)(symbol.Symbol)
_register_op_map(opns.GET_VALID_COUNT)(symbol.Symbol)
_register_op_map(opns.NON_MAX_SUPRESSION)(symbol.Symbol)
_register_op_map(opns.CEIL)(symbol.Symbol)
_register_op_map(opns.RIGHT_SHIFT)(symbol.Symbol)
_register_op_map(opns.AS_TYPE)(symbol.Symbol)
_register_op_map(opns.ADV_INDEX)(symbol.Symbol)
_register_op_map(opns.CALL_TIR)(symbol.Symbol)
_register_op_map(opns.CALL_DPS_PACKED)(symbol.Symbol)
_register_op_map(opns.ADD)(symbol.Symbol)
_register_op_map(opns.SUB)(symbol.Symbol)
_register_op_map(opns.MATMUL)(symbol.Symbol)
_register_op_map(opns.DIV)(symbol.Symbol)
_register_op_map(opns.NEGATIVE)(symbol.Symbol)
_register_op_map(opns.ABS)(symbol.Symbol)
_register_op_map(opns.LOG)(symbol.Symbol)
_register_op_map(opns.SQRT)(symbol.Symbol)
_register_op_map(opns.POW)(symbol.Symbol)
_register_op_map(opns.PASS)(symbol.Symbol)
_register_op_map(opns.ARANGE)(symbol.Symbol)
_register_op_map(opns.ZEROS_LIKE)(symbol.Symbol)
_register_op_map(opns.ONES_LIKE)(symbol.Symbol)
_register_op_map(opns.IF)(symbol.Symbol)
_register_op_map(opns.ARGWHERE)(symbol.Symbol)
_register_op_map(opns.REQUANT)(symbol.Symbol)
_register_op_map(opns.PCLIP)(symbol.Symbol)
_register_op_map(opns.RS_PCLIP)(symbol.Symbol)
_register_op_map(opns.LUT)(symbol.Symbol)
