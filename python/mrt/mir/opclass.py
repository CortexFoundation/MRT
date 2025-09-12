import typing
import numpy as np
from dataclasses import dataclass
from . import opns
from . import symbol

MRT_OP_MAP: typing.Dict[str, typing.Any] = {}

#def _register_op_map_(op_name: str, clss:typing.Any=None):
#    if len(op_name)>0 and clss!=None:
#        if op_name not in MRT_OP_MAP:
#            MRT_OP_MAP[op_name] = clss
#    return MRT_OP_MAP

def _register_op_map(op_name: str): #, clss:typing.Any=None):
    def _wrapper(clss: typing.Any=None):
        if len(op_name)>0 and clss!=None:
            if op_name not in MRT_OP_MAP:
                MRT_OP_MAP[op_name] = clss
        return clss
    return _wrapper

@dataclass(init=False)
class Conv2D(symbol.Symbol):

    op_name = opns.CONV2D

    @property
    def strides(self) -> typing.Tuple[int, int]:
        default_val = (1,1)
        return self._strides if self._strides else self.attrs['strides'] if 'strides' in self.attrs else default_val

    @property
    def padding(self) -> typing.Tuple[int, int, int, int]:
        default_val = (0,0,0,0)
        return self._padding if self._padding else self.attrs['padding'] if 'padding' in self.attrs else default_val

    @property
    def dilation(self) -> typing.Tuple[int, int]:
        default_val = (1,1)
        return self._dilation if self._dilation else self.attrs['dilation'] if 'dilation' in self.attrs else default_val

    @property
    def kernel_size(self) -> typing.Tuple[int, int]:
        default_val = (3,3)
        return self._kernel_size if self._kernel_size else self.attrs['kernel_size'] if 'kernel_size' in self.attrs else default_val

    def __init__(self, name_or_inst: typing.Union[str, symbol.Symbol], **kwargs):
        assert isinstance(name_or_inst, str) or isinstance(name_or_inst, symbol.Symbol)
        if isinstance(name_or_inst, str):
            self.name = name_or_inst
            self.args = kwargs.pop('args', [])
            self.attrs = kwargs.pop('attrs', {})
            self.extra_attrs = {}
        else:
            # clone mode
            self.name = name_or_inst.name
            self.args = [a for a in name_or_inst.args]
            self.attrs = {k: v for k, v in name_or_inst.attrs.items()}
            self.extra_attrs = {k: v for k, v in name_or_inst.extra_attrs.items()}

        # TODO: what if strides not in attrs?
        if 'strides' in self.attrs:
            self._strides = self.attrs['strides']
        if 'padding' in self.attrs:
            self._padding = self.attrs['padding']
        if 'dilation' in self.attrs:
            self._dilation = self.attrs['dilation']
        if 'kernel_size' in self.attrs:
            self._kernel_size = self.attrs['kernel_size']


@dataclass(init=False)
class Dropout(symbol.Symbol):

    op_name = opns.DROP_OUT

    @property
    def rate(self) -> float:
        default_val = 0.0
        return self._rate if self._rate else self.attrs['rate'] if 'rate' in self.attrs else default_val 
    
    def __init__(self, name:str, **kwargs):
        self.name = name
        self.args = kwargs.pop('args', [])
        self.attrs = kwargs.pop('attrs', {})
        self.extra_attrs = {}

        self._rate = self.attrs['rate']

@dataclass(init=False)
class Clip(symbol.Symbol):

    op_name = opns.CLIP
    
    @property
    def min(self) -> float:
        default_val = np.nan
        return self._min if self._min else self.attrs['min'] if 'min' in self.attrs else default_val

    @property
    def max(self) -> float:
        default_val = np.nan
        return self._max if self._max else self.attrs['max'] if 'max' in self.attrs else default_val

    def __init__(self, name:str, **kwargs):
        self.name = name
        self.args = kwargs.pop('args', [])
        self.attrs = kwargs.pop('attrs', {})
        self.extra_attrs = {}

        self._min = self.attrs['min']
        self._max = self.attrs['max']


@dataclass(init=False)
class BatchNorm(symbol.Symbol):

    op_name = opns.BATCH_NORM

    @property
    def axis(self) -> float:
        default_val = 1
        return self._axis if self._axis else self.attrs['axis'] if 'axis' in self.attrs else default_val

    @property
    def epsilon(self) -> float:
        default_val = 1e-5
        return self._epsilon if self._epsilon else self.attrs['epsilon'] if 'epsilon' in self.attrs else default_val

    @property
    def center(self) -> float:
        default_val = True
        return self._center if self._center else self.attrs['center'] if 'center' in self.attrs else default_val

    @property
    def scale(self) -> float:
        default_val = True
        return self._scale if self._scale else self.attrs['scale'] if 'scale' in self.attrs else default_val

    def __init__(self, name:str, **kwargs):
        self.name = name
        self.args = kwargs.pop('args', [])
        self.attrs = kwargs.pop('attrs', {})
        self.extra_attrs = {}

        self._axis = self.attrs['axis']
        self._epsilon = self.attrs['epsilon']
        self._center = self.attrs['center']
        self._scale = self.attrs['scale']

@dataclass(init=False)
class Dense(symbol.Symbol):

    op_name = opns.DENSE
    
    def __init__(self, name:str, **kwargs):
        self.name = name
        self.args = kwargs.pop('args', [])
        self.attrs = kwargs.pop('attrs', {})
        self.extra_attrs = {}
 
@dataclass(init=False)
class TupleGetItem(symbol.Symbol):

    op_name = opns.TUPLE_GET_ITEM
    
    @property
    def index(self) -> float:
        default_val = 0
        return self._index if self._index else self.attrs['index'] if 'index' in self.attrs else default_val

    def __init__(self, name:str, **kwargs):
        self.name = name
        self.args = kwargs.pop('args', [])
        self.attrs = kwargs.pop('attrs', {})
        self.extra_attrs = {}

        self._index  = self.attrs['index']

@dataclass(init=False)
class Multiply(symbol.Symbol):

    op_name = opns.MUL
    
    def __init__(self, name:str, **kwargs):
        self.name = name
        self.args = kwargs.pop('args', [])
        self.attrs = kwargs.pop('attrs', {})
        self.extra_attrs = {}
 
_register_op_map(opns.CONV2D)(Conv2D)
_register_op_map(opns.DROP_OUT)(Dropout)
_register_op_map(opns.CLIP)(Clip)
_register_op_map(opns.BATCH_NORM)(BatchNorm)
_register_op_map(opns.DENSE)(Dense)
_register_op_map(opns.TUPLE_GET_ITEM)(TupleGetItem)
_register_op_map(opns.MUL)(Multiply)
