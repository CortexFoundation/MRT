import typing
import numpy as np
from dataclasses import dataclass

from mrt.common.utils import N
from . import opns
from . import symbol
from .symbol import SelfSymbol

#SelfSymbol = typing.TypeVar("SelfSymbol", bound="Symbol")

SymbolCreator = typing.Union[typing.Callable[[typing.Any, typing.Any], typing.Type[symbol.Symbol]], SelfSymbol]
#SymbolCreator = typing.Union[typing.Callable[[...], symbol.Symbol], SelfSymbol]

MRT_OP_MAP: typing.Dict[str, SymbolCreator] = {}

def _register_op_map(op_name: str):
    def _wrapper(clss: SymbolCreator = None) -> SymbolCreator:
        if len(op_name) > 0 and clss != None:
            if op_name not in MRT_OP_MAP:
                MRT_OP_MAP[op_name] = clss
            else:
                print(f'Warning: "{op_name}" Alreary Registered In MRT_OP_MAP, IsBeing Overrided!')
                MRT_OP_MAP[op_name] = clss
        return clss
    return _wrapper


# OPs from external (not in MRT op), using custom op_name with default op_func
#y = extern_opfunc("tanh")(X)
def extern_opfunc(op_name: str):
    def op_func(*args, **attrs):
        return symbol.Symbol(*args, op_name=op_name, **attrs)
    return op_func


def _from_dict_attrs(cls, d: dict, attr_keys:typing.List[str]=[], **kwargs):
    data = cls.default_dict()
    data.update(d)
    data.update(kwargs)
    data = cls.update_dict(data)
    basedata = {k: data[k] for k in data if k in ['name', 'op_name', 'extra_attrs']}
    attrsdata = {k: data['attrs'][k] for k in data['attrs'] if k in attr_keys}
    try:
        out = cls(*data['args'], **attrsdata, **basedata)
    except Exception as e:
        raise e
    return out

# OPs without attrs, just register function (funcName should be lower case)
def var(name=None, op_name=None, shape=(), dtype=float) -> symbol.Symbol:
    op_name = op_name or opns.VAR
    assert op_name == opns.VAR
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[], attrs={}, extra_attrs={'shape': shape or (), 'dtype': dtype or float})

#def _return_func_single_arg(op_name: op_name):
def relu(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.RELU
    assert op_name == opns.RELU
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def silu(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.SILU
    assert op_name == opns.SILU
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

 
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

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['strides', 'padding', 'groups', 'dilation', 'kernel_size'], **kwargs)

@dataclass(init=False)
class Dropout(symbol.Symbol):
    op_name = opns.DROP_OUT

    @property
    def p(self) -> float:
        default_val = 0.5
        return self.attrs['p'] if 'p' in self.attrs else default_val
    
    def __init__(self, X, name=None, op_name=None, p:float = 0.5, extra_attrs=None):
        op_name = op_name or opns.DROP_OUT
        assert op_name == opns.DROP_OUT
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'p': p}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['p'], **kwargs)

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
        return _from_dict_attrs(cls, d, ['min', 'max'], **kwargs)


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
    def momentum(self) -> float:
        default_val = 0.1
        return self.attrs['center'] if 'center' in self.attrs else default_val

    def __init__(self, X, Gamma, Beta, Mean, Var, name=None, op_name=None, axis:int = 1, epsilon:float = 1e-5, momentum:float = 0.1, extra_attrs=None):
        op_name = op_name or opns.BATCH_NORM
        assert op_name == opns.BATCH_NORM
        super().__init__(name=name or N.n(), op_name=op_name, args=[X, Gamma, Beta, Mean, Var], attrs={'axis': axis, 'epsilon': epsilon, 'momentum': momentum}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['axis', 'epsilon', 'momentum'], **kwargs)

 
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
        return _from_dict_attrs(cls, d, ['index'], **kwargs)


@dataclass(init=False)
class LeakyRelu(symbol.Symbol):
    op_name = opns.LEAKY_RELU

    @property
    def negative_slope(self) -> float:
        default_val = 1e-2
        return self.attrs['negative_slope'] if 'negative_slope' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, negative_slope:float = 1e-2, extra_attrs=None):
        op_name = op_name or opns.LEAKY_RELU
        assert op_name == opns.LEAKY_RELU
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'negative_slope': negative_slope}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['negative_slope'], **kwargs)


def dense(X, W, B, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.DENSE
    assert op_name == opns.DENSE
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X, W, B], attrs={}, extra_attrs=extra_attrs or {})

@dataclass(init=False)
class Hardtanh(symbol.Symbol):
    op_name = opns.HARDTANH

    @property
    def min_val(self) -> float:
        default_val = -1.0
        return self.attrs['min_val'] if 'min_val' in self.attrs else default_val

    @property
    def max_val(self) -> float:
        default_val = 1.0
        return self.attrs['max_val'] if 'max_val' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, min_val:float = -1.0, max_val:float = 1.0, extra_attrs=None):
        op_name = op_name or opns.HARDTANH
        assert op_name == opns.HARDTANH
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'min_val': min_val, 'max_val':max_val}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['min_val', 'max_val'], **kwargs)


@dataclass(init=False)
class AdaptiveAvgPool2D(symbol.Symbol):
    op_name = opns.ADAPTIVE_AVG_POOL2D

    @property
    def output_size(self) -> typing.Union[int, typing.Tuple[int, int]]:
        default_val = 0
        return self.attrs['output_size'] if 'output_size' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, output_size:typing.Union[int, typing.Tuple[int, int]]=0, extra_attrs=None):
        op_name = op_name or opns.ADAPTIVE_AVG_POOL2D
        assert op_name == opns.ADAPTIVE_AVG_POOL2D
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'output_size': output_size}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['output_size'], **kwargs)

@dataclass(init=False)
class AvgPool2D(symbol.Symbol):
    op_name = opns.AVG_POOL2D

    @property
    def pool_size(self) -> typing.Tuple[int, int]:
        default_val = (2, 2)
        return self.attrs['pool_size'] if 'pool_size' in self.attrs else default_val
    @property
    def strides(self):
        default_val = None
        return self.attrs['strides'] if 'strides' in self.attrs else default_val
    @property
    def padding(self) -> int:
        default_val = 0
        return self.attrs['padding'] if 'padding' in self.attrs else default_val
    @property
    def ceil_mode(self) -> bool:
        default_val = False
        return self.attrs['ceil_mode'] if 'ceil_mode' in self.attrs else default_val
    @property
    def layout(self) -> str:
        default_val = 'NCHW'
        return self.attrs['layout'] if 'layout' in self.attrs else default_val
    @property
    def count_include_pad(self) -> bool:
        default_val = True
        return self.attrs['count_include_pad'] if 'count_include_pad' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, pool_size=(2,2), strides=None, padding=0, ceil_mode=False, layout='NCHW', count_include_pad=True, extra_attrs=None):
        op_name = op_name or opns.AVG_POOL2D
        assert op_name == opns.AVG_POOL2D
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'pool_size':pool_size, 'strides':strides, 'padding':padding, 'ceil_mode':ceil_mode, 'layout':layout, 'count_include_pad':count_include_pad}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['pool_size', 'strides', 'padding', 'ceil_mode', 'layout', 'count_include_pad'], **kwargs)

@dataclass(init=False)
class MaxPool2D(symbol.Symbol):
    op_name = opns.MAX_POOL2D

    @property
    def pool_size(self) -> typing.Tuple[int, int]:
        default_val = (2, 2)
        return self.attrs['pool_size'] if 'pool_size' in self.attrs else default_val
    @property
    def layout(self) -> str:
        default_val = 'NCHW'
        return self.attrs['layout'] if 'layout' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, pool_size=(2,2), layout='NCHW', extra_attrs=None):
        op_name = op_name or opns.MAX_POOL2D
        assert op_name == opns.MAX_POOL2D
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'pool_size':pool_size, 'layout':layout}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['pool_size', 'layout'], **kwargs)


@dataclass(init=False)
class Softmax(symbol.Symbol):
    op_name = opns.SOFTMAX

    @property
    def axis(self) -> typing.Optional[int]:
        default_val = None
        return self.attrs['axis'] if 'axis' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, axis=None, extra_attrs=None):
        op_name = op_name or opns.SOFTMAX
        assert op_name == opns.SOFTMAX
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'axis':axis}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['axis'], **kwargs)


@dataclass(init=False)
class LogSoftmax(symbol.Symbol):
    op_name = opns.LOG_SOFTMAX

    @property
    def axis(self) -> typing.Optional[int]:
        default_val = None
        return self.attrs['axis'] if 'axis' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, axis=None, extra_attrs=None):
        op_name = op_name or opns.LOG_SOFTMAX
        assert op_name == opns.LOG_SOFTMAX
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'axis':axis}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['axis'], **kwargs)


def exp(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.EXP
    assert op_name == opns.EXP
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def sigmoid(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.SIGMOID
    assert op_name == opns.SIGMOID
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

@dataclass(init=False)
class Sum(symbol.Symbol):
    op_name = opns.SUM

    @property
    def dim(self) -> typing.Optional[typing.Tuple[int, ...]]:
        default_val = None
        return self.attrs['dim'] if 'dim' in self.attrs else default_val

    @property
    def keepdim(self) -> typing.Optional[bool]:
        default_val = None
        return self.attrs['keepdim'] if 'keepdim' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, dim=None, keepdim=None, extra_attrs=None):
        op_name = op_name or opns.SUM
        assert op_name == opns.SUM
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'dim': dim, 'keepdim': keepdim}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim', 'keepdim'], **kwargs)


@dataclass(init=False)
class Mean(symbol.Symbol):
    op_name = opns.MEAN

    @property
    def dim(self) -> typing.Optional[typing.Tuple[int, ...]]:
        default_val = None
        return self.attrs['dim'] if 'dim' in self.attrs else default_val

    @property
    def keepdim(self) -> typing.Optional[bool]:
        default_val = None
        return self.attrs['keepdim'] if 'keepdim' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, dim=None, keepdim=None, extra_attrs=None):
        op_name = op_name or opns.MEAN
        assert op_name == opns.MEAN
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'dim': dim, 'keepdim': keepdim}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim', 'keepdim'], **kwargs)


@dataclass(init=False)
class MaxAxis(symbol.Symbol):
    op_name = opns.MAX_AXIS

    @property
    def dim(self) -> typing.Optional[typing.Tuple[int, ...]]:
        default_val = None
        return self.attrs['dim'] if 'dim' in self.attrs else default_val

    @property
    def keepdim(self) -> typing.Optional[bool]:
        default_val = None
        return self.attrs['keepdim'] if 'keepdim' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, dim=None, keepdim=None, extra_attrs=None):
        op_name = op_name or opns.MAX_AXIS
        assert op_name == opns.MAX_AXIS
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'dim': dim, 'keepdim': keepdim}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim', 'keepdim'], **kwargs)

def maximum(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.MAXIMUM
    assert op_name == opns.MAXIMUM
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def minimum(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.MINIMUM
    assert op_name == opns.MINIMUM
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def repeat(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.REPEAT
    assert op_name == opns.REPEAT
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

@dataclass(init=False)
class Squeeze(symbol.Symbol):
    op_name = opns.SQUEEZE

    @property
    def dim(self) -> typing.Optional[int]:
        default_val = None
        return self.attrs['dim'] if 'dim' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, dim=None, extra_attrs=None):
        op_name = op_name or opns.SQUEEZE
        assert op_name == opns.SQUEEZE
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'dim': dim}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim'], **kwargs)

@dataclass(init=False)
class Flatten(symbol.Symbol):
    op_name = opns.FLATTEN

    @property
    def start_dim(self) -> int:
        default_val = 0
        return self.attrs['start_dim'] if 'start_dim' in self.attrs else default_val

    @property
    def end_dim(self) -> int:
        default_val = -1
        return self.attrs['end_dim'] if 'end_dim' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, start_dim=0, end_dim=-1, extra_attrs=None):
        op_name = op_name or opns.FLATTEN
        assert op_name == opns.FLATTEN
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'start_dim': start_dim, 'end_dim':end_dim}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['start_dim', 'end_dim'], **kwargs)


@dataclass(init=False)
class Reshape(symbol.Symbol):
    op_name = opns.RESHAPE

    @property
    def newshape(self) -> typing.Tuple[int,...]:
        default_val = None
        return self.attrs['newshape'] if 'newshape' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, newshape=None, extra_attrs=None):
        op_name = op_name or opns.RESHAPE
        assert op_name == opns.RESHAPE
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'newshape': newshape}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['newshape'], **kwargs)


@dataclass(init=False)
class Concat(symbol.Symbol):
    op_name = opns.CONCAT

    @property
    def axis(self) -> int:
        default_val = 0
        return self.attrs['axis'] if 'axis' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, axis=None, extra_attrs=None):
        op_name = op_name or opns.CONCAT
        assert op_name == opns.CONCAT
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'axis': axis}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['axis'], **kwargs)


@dataclass(init=False)
class Split(symbol.Symbol):
    op_name = opns.SPLIT

    @property
    def split_size(self) -> typing.List[int]:
        default_val = []
        return self.attrs['split_size'] if 'split_size' in self.attrs else default_val

    @property
    def dim(self) -> int:
        default_val = 0
        return self.attrs['dim'] if 'dim' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, split_size=[], dim=0, extra_attrs=None):
        op_name = op_name or opns.SPLIT
        assert op_name == opns.SPLIT
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'split_size': split_size, 'dim': dim}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['split_size', 'dim'], **kwargs)


@dataclass(init=False)
class Transpose(symbol.Symbol):
    op_name = opns.TRANSPOSE

    @property
    def dim0(self) -> int:
        default_val = 0
        return self.attrs['dim0'] if 'dim0' in self.attrs else default_val

    @property
    def dim1(self) -> int:
        default_val = 0
        return self.attrs['dim1'] if 'dim1' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, dim0=0, dim1=0, extra_attrs=None):
        op_name = op_name or opns.TRANSPOSE
        assert op_name == opns.TRANSPOSE
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'dim0': dim0, 'dim1': dim1}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim0', 'dim1'], **kwargs)


@dataclass(init=False)
class BroadcastTo(symbol.Symbol):
    op_name = opns.BROADCAST_TO

    @property
    def newshape(self) -> typing.Tuple[int,...]:
        default_val = None
        return self.attrs['newshape'] if 'newshape' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, newshape=None, extra_attrs=None):
        op_name = op_name or opns.BROADCAST_TO
        assert op_name == opns.BROADCAST_TO
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'newshape': newshape}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['newshape'], **kwargs)


@dataclass(init=False)
class ExpandDims(symbol.Symbol):
    op_name = opns.EXPAND_DIMS

    @property
    def newshape(self) -> typing.Tuple[int,...]:
        default_val = None
        return self.attrs['newshape'] if 'newshape' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, newshape=None, extra_attrs=None):
        op_name = op_name or opns.EXPAND_DIMS
        assert op_name == opns.EXPAND_DIMS
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'newshape': newshape}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['newshape'], **kwargs)


@dataclass(init=False)
class Tile(symbol.Symbol):
    op_name = opns.TILE

    @property
    def dims(self) -> typing.Tuple[int,...]:
        default_val = None
        return self.attrs['dims'] if 'dims' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, dims=None, extra_attrs=None):
        op_name = op_name or opns.TILE
        assert op_name == opns.TILE
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'dims': dims}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dims'], **kwargs)

def where(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.WHERE
    assert op_name == opns.WHERE
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def greater(X, Y, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.GREATER
    assert op_name == opns.GREATER
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X, Y], attrs={}, extra_attrs=extra_attrs or {})

@dataclass(init=False)
class NonMaxSuppression(symbol.Symbol):
    op_name = opns.NON_MAX_SUPRESSION

    @property
    def iou_threshold(self) -> float:
        default_val = 0.5
        return self.attrs['iou_threshold'] if 'iou_threshold' in self.attrs else default_val
    @property
    def score_threshold(self) -> typing.Optional[float]:
        default_val = None
        return self.attrs['score_threshold'] if 'score_threshold' in self.attrs else default_val

    def __init__(self, X, name=None, op_name=None, iou_threshold=0.5, score_threshold=None, extra_attrs=None):
        op_name = op_name or opns.NON_MAX_SUPRESSION
        assert op_name == opns.NON_MAX_SUPRESSION
        super().__init__(name=name or N.n(), op_name=op_name, args=[X], attrs={'iou_threshold': iou_threshold,'score_threshold':score_threshold}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dims'], **kwargs)


def ceil(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.CEIL
    assert op_name == opns.CEIL
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def rightShift(X, Y, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.RIGHT_SHIFT
    assert op_name == opns.RIGHT_SHIFT
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X, Y], attrs={}, extra_attrs=extra_attrs or {})

@dataclass(init=False)
class Add(symbol.Symbol):
    op_name = opns.ADD

    @property
    def alpha(self) -> int:
        default_val = 1
        return self.attrs['alpha'] if 'alpha' in self.attrs else default_val

    def __init__(self, X, Y, name=None, op_name=None, alpha=1, extra_attrs=None):
        op_name = op_name or opns.ADD
        assert op_name == opns.ADD
        super().__init__(name=name or N.n(), op_name=op_name, args=[X, Y], attrs={'alpha': alpha}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['alpha'], **kwargs)

@dataclass(init=False)
class Sub(symbol.Symbol):
    op_name = opns.SUB

    @property
    def alpha(self) -> int:
        default_val = 1
        return self.attrs['alpha'] if 'alpha' in self.attrs else default_val

    def __init__(self, X, Y, name=None, op_name=None, alpha=1, extra_attrs=None):
        op_name = op_name or opns.SUB
        assert op_name == opns.SUB
        super().__init__(name=name or N.n(), op_name=op_name, args=[X, Y], attrs={'alpha': alpha}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['alpha'], **kwargs)

def mul(X, Y, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.MUL
    assert op_name == opns.MUL
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X, Y], attrs={}, extra_attrs=extra_attrs or {})

def matMul(X, Y, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.MATMUL
    assert op_name == opns.MATMUL
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X, Y], attrs={}, extra_attrs=extra_attrs or {})

@dataclass(init=False)
class Div(symbol.Symbol):
    op_name = opns.DIV

    @property
    def rounding_mode(self) -> typing.Optional[str]:
        default_val = None
        return self.attrs['rounding_mode'] if 'rounding_mode' in self.attrs else default_val

    def __init__(self, X, Y, name=None, op_name=None, rounding_mode=None, extra_attrs=None):
        op_name = op_name or opns.DIV
        assert op_name == opns.DIV
        super().__init__(name=name or N.n(), op_name=op_name, args=[X, Y], attrs={'rounding_mode': rounding_mode}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['rounding_mode'], **kwargs)

def negative(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.NEGATIVE
    assert op_name == opns.NEGATIVE
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def abs(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.ABS
    assert op_name == opns.ABS
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def log(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.LOG
    assert op_name == opns.LOG
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def sqrt(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.SQRT
    assert op_name == opns.SQRT
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def pow(X, Y, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.POW
    assert op_name == opns.POW
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X, Y], attrs={}, extra_attrs=extra_attrs or {})

def pass_(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.PASS
    assert op_name == opns.PASS
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

@dataclass(init=False)
class Arange(symbol.Symbol):
    op_name = opns.ARANGE

    @property
    def end(self) -> int:
        default_val = 0
        return self.attrs['end'] if 'end' in self.attrs else default_val

    @property
    def start(self) -> int:
        default_val = 0
        return self.attrs['start'] if 'start' in self.attrs else default_val

    @property
    def step(self) -> int:
        default_val = 1
        return self.attrs['step'] if 'step' in self.attrs else default_val

    def __init__(self, name=None, op_name=None, end=0, start=0, step=1, extra_attrs=None):
        op_name = op_name or opns.ARANGE
        assert op_name == opns.ARANGE
        super().__init__(name=name or N.n(), op_name=op_name, args=[], attrs={'end': end, 'start': start, 'step': step}, extra_attrs=extra_attrs or {})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['end', 'start', 'step'], **kwargs)

def zerosLike(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.ZEROS_LIKE
    assert op_name == opns.ZEROS_LIKE
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})

def onesLike(X, name=None, op_name=None, extra_attrs=None) -> symbol.Symbol:
    op_name = op_name or opns.ONES_LIKE
    assert op_name == opns.ONES_LIKE
    return symbol.Symbol(name=name or N.n(), op_name=op_name, args=[X], attrs={}, extra_attrs=extra_attrs or {})


_register_op_map(opns.VAR)(var)
_register_op_map(opns.RELU)(relu)

_register_op_map(opns.CONV2D)(Conv2D)
_register_op_map(opns.DROP_OUT)(Dropout)
_register_op_map(opns.CLIP)(Clip)
_register_op_map(opns.BATCH_NORM)(BatchNorm)
_register_op_map(opns.TUPLE_GET_ITEM)(TupleGetItem)

_register_op_map(opns.LEAKY_RELU)(LeakyRelu)

_register_op_map(opns.MUL)(mul)
_register_op_map(opns.DENSE)(dense)
_register_op_map(opns.HARDTANH)(Hardtanh)
_register_op_map(opns.SILU)(silu)
_register_op_map(opns.ADAPTIVE_AVG_POOL2D)(AdaptiveAvgPool2D)
_register_op_map(opns.AVG_POOL2D)(AvgPool2D)
_register_op_map(opns.MAX_POOL2D)(MaxPool2D)
_register_op_map(opns.SOFTMAX)(Softmax)
_register_op_map(opns.LOG_SOFTMAX)(LogSoftmax)
_register_op_map(opns.EXP)(exp)
_register_op_map(opns.SIGMOID)(sigmoid)
_register_op_map(opns.SUM)(Sum)
_register_op_map(opns.MEAN)(Mean)
_register_op_map(opns.MAX_AXIS)(MaxAxis)
_register_op_map(opns.MAXIMUM)(maximum)
_register_op_map(opns.MINIMUM)(minimum)


_register_op_map(opns.REPEAT)(repeat)
_register_op_map(opns.SQUEEZE)(Squeeze)
_register_op_map(opns.FLATTEN)(Flatten)
_register_op_map(opns.RESHAPE)(Reshape)
_register_op_map(opns.CONCAT)(Concat)
_register_op_map(opns.SPLIT)(Split)
_register_op_map(opns.TRANSPOSE)(Transpose)
_register_op_map(opns.BROADCAST_TO)(BroadcastTo)
_register_op_map(opns.EXPAND_DIMS)(ExpandDims)
_register_op_map(opns.TILE)(Tile)
_register_op_map(opns.WHERE)(where)
_register_op_map(opns.GREATER)(greater)
_register_op_map(opns.NON_MAX_SUPRESSION)(NonMaxSuppression)

_register_op_map(opns.CEIL)(ceil)
_register_op_map(opns.RIGHT_SHIFT)(rightShift)

_register_op_map(opns.ADD)(Add)
_register_op_map(opns.SUB)(Sub)
_register_op_map(opns.MATMUL)(matMul)
_register_op_map(opns.DIV)(Div)
_register_op_map(opns.NEGATIVE)(negative)
_register_op_map(opns.ABS)(abs)
_register_op_map(opns.LOG)(log)
_register_op_map(opns.SQRT)(sqrt)
_register_op_map(opns.POW)(pow)
_register_op_map(opns.PASS)(pass_)
_register_op_map(opns.ARANGE)(Arange)
_register_op_map(opns.ZEROS_LIKE)(zerosLike)
_register_op_map(opns.ONES_LIKE)(onesLike)


# Add default register Class for MRT OP Not Implemented!
_register_op_map(opns.TUPLE)(extern_opfunc(opns.TUPLE))
_register_op_map(opns.AS_TYPE)(extern_opfunc(opns.AS_TYPE))
_register_op_map(opns.ADV_INDEX)(extern_opfunc(opns.ADV_INDEX))
_register_op_map(opns.CALL_TIR)(extern_opfunc(opns.CALL_TIR))
_register_op_map(opns.CALL_DPS_PACKED)(extern_opfunc(opns.CALL_DPS_PACKED))

_register_op_map(opns.IF)(symbol.Symbol)
_register_op_map(opns.ARGWHERE)(symbol.Symbol)
_register_op_map(opns.REQUANT)(symbol.Symbol)
_register_op_map(opns.PCLIP)(symbol.Symbol)
_register_op_map(opns.RS_PCLIP)(symbol.Symbol)
_register_op_map(opns.LUT)(symbol.Symbol)

_register_op_map(opns.BATCH_FLATTEN)(symbol.Symbol)
_register_op_map(opns.STRIDED_SLICE)(symbol.Symbol)
_register_op_map(opns.SLICE_LIKE)(symbol.Symbol)
_register_op_map(opns.GET_VALID_COUNT)(symbol.Symbol)

