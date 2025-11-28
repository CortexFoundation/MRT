import typing
import numpy as np

from mrt.common.utils import N
from . import opns
from . import symbol

SymbolCreator = typing.Union[typing.Callable[[typing.Any, ...], typing.Type[symbol.Symbol]], symbol.SelfSymbol]

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
# y = extern_opfunc("tanh")(X)
def extern_opfunc(op_name: str):
    def op_func(*args, name=None, extra_attrs=None, **kwargs):
        return symbol.Symbol(*args, name=name or N.n(), op_name=op_name, extra_attrs=extra_attrs or {}, **kwargs)
    return op_func

def _from_dict_attrs(cls, d: dict, attr_keys:typing.List[str]=[], **kwargs):
    data = cls.default_dict()
    data.update(d)
    data.update(kwargs)
    data = cls.update_dict(data)
    basedata = {k: data[k] for k in data if k in ['name', 'extra_attrs']}
    attrsdata = {k: data['attrs'][k] for k in data['attrs'] if k in attr_keys}
    try:
        out = cls(*data['args'], **basedata, **attrsdata)
    except Exception as e:
        raise e
    return out

# OPs without attrs, just register function (funcName should be lower case)
def var(name=None, shape=(), dtype=float) -> symbol.Symbol:
    return symbol.Symbol(name=name or N.n(), op_name=opns.VAR, extra_attrs={'shape': shape or (), 'dtype': dtype or float})

#def _return_func_single_arg(op_name: op_name):
def relu(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.RELU, extra_attrs=extra_attrs or {})

def silu(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.SILU, extra_attrs=extra_attrs or {})

 
class Conv2D(symbol.Symbol):
    op_name = opns.CONV2D

    @property
    def strides(self) -> typing.Tuple[int, int]:
        return self.attrs['strides']

    @property
    def padding(self) -> typing.Tuple[int, int, int, int]:
        return self.attrs['padding']

    @property
    def groups(self) -> int:
        return self.attrs['groups']

    @property
    def dilation(self) -> typing.Tuple[int, int]:
        return self.attrs['dilation']


    # Follows (*args, name, **attrs)
    def __init__(self, X, W, name=None, strides=(1,1), padding=(0,0,0,0), groups=1, dilation=(1,1), kernel_layout='OIHW', extra_attrs=None):
        assert len(W.shape) == 4, f'Wrong Weight Shape for Conv2D: {W.shape}'
        kernel_size = (W.shape[2], W.shape[3])
        #attrs = {'strides':strides, 'padding':padding, 'groups':groups, 'dilation':dilation, 'kernel_size':kernel_size, 'kernel_layout': kernel_layout}
        attrs = {'strides':strides, 'padding':padding, 'groups':groups, 'dilation':dilation}
        super().__init__(X, W, name=name or N.n(), op_name=opns.CONV2D, extra_attrs=extra_attrs or {}, **attrs)

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        # Auto inferred 'kernel_size'
        return _from_dict_attrs(cls, d, ['strides', 'padding', 'groups', 'dilation'], **kwargs)

def conv2d(*args, **kwargs):
    #def conv2d(X, W, name=None, op_name=None, strides=(1,1), padding=(0,0,0,0), groups=1, dilation=(1,1), kernel_layout='OIHW', extra_attrs=None):
    return Conv2D(*args, **kwargs) #X, W, name, op_name, strides, padding, groups, dilation, kernel_layout, extra_attrs)


class Dropout(symbol.Symbol):
    op_name = opns.DROP_OUT

    @property
    def p(self) -> float:
        return self.attrs['p']
    
    def __init__(self, X, name=None, p:float = 0.5, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.DROP_OUT, extra_attrs=extra_attrs or {}, **{'p': p})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['p'], **kwargs)

def dropout(*args, **kwargs):
    return dropout(*args, **kwargs)


class Clip(symbol.Symbol):
    op_name = opns.CLIP
    
    @property
    def min(self) -> float:
        assert 'min' in self.attrs
        return self.attrs['min']

    @property
    def max(self) -> float:
        assert 'max' in self.attrs
        return self.attrs['max']

    def __init__(self, X, name=None, min:float = np.nan, max:float = np.nan, extra_attrs=None):
        assert min != np.nan
        assert max != np.nan
        super().__init__(X, name=name or N.n(), op_name=opns.CLIP, extra_attrs=extra_attrs or {}, **{'min': min, 'max': max})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['min', 'max'], **kwargs)

def clip(*args, **kwargs):
    return Clip(*args, **kwargs)

class BatchNorm(symbol.Symbol):
    op_name = opns.BATCH_NORM

    @property
    def axis(self) -> int:
        return self.attrs['axis']

    @property
    def epsilon(self) -> float:
        return self.attrs['epsilon']

    @property
    def momentum(self) -> float:
        return self.attrs['momentum']

    @property
    def center(self) -> bool:
        return self.attrs['center']

    @property
    def scale(self) -> bool:
        return self.attrs['scale']

    def __init__(self, X, Gamma, Beta, Mean, Var, name=None, axis:int = 1, epsilon:float = 1e-5, momentum:float = 0.1, center=True, scale=True, extra_attrs=None):
        super().__init__(*[X, Gamma, Beta, Mean, Var], name=name or N.n(), op_name=opns.BATCH_NORM, extra_attrs=extra_attrs or {}, **{'axis': axis, 'epsilon': epsilon, 'momentum': momentum, 'center': center, 'scale': scale})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['axis', 'epsilon', 'momentum', 'center', 'scale'], **kwargs)

def batch_norm(*args, **kwargs):
    return BatchNorm(*args, **kwargs)

 
class TupleGetItem(symbol.Symbol):
    op_name = opns.TUPLE_GET_ITEM
    
    @property
    def index(self) -> float:
        return self.attrs['index']

    def __init__(self, X, name=None, index:int = 0, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.TUPLE_GET_ITEM, extra_attrs=extra_attrs or {}, **{'index': index})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['index'], **kwargs)

def tuple_get_item(*args, **kwargs):
    return TupleGetItem(*args, **kwargs)


class LeakyRelu(symbol.Symbol):
    op_name = opns.LEAKY_RELU

    @property
    def negative_slope(self) -> float:
        return self.attrs['negative_slope']

    def __init__(self, X, name=None, negative_slope:float = 1e-2, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.LEAKY_RELU, extra_attrs=extra_attrs or {}, **{'negative_slope': negative_slope})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['negative_slope'], **kwargs)

def leaky_relu(*args, **kwargs):
    return LeakyRelu(*args, **kwargs)


def dense(X, W, B, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(*[X, W, B], name=name or N.n(), op_name=opns.DENSE, extra_attrs=extra_attrs or {})

class Hardtanh(symbol.Symbol):
    op_name = opns.HARDTANH

    @property
    def min_val(self) -> float:
        return self.attrs['min_val']

    @property
    def max_val(self) -> float:
        return self.attrs['max_val']

    def __init__(self, X, name=None, min_val:float = -1.0, max_val:float = 1.0, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.HARDTANH, extra_attrs=extra_attrs or {}, **{'min_val': min_val, 'max_val':max_val})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['min_val', 'max_val'], **kwargs)

def hard_tanh(*args, **kwargs):
    return Hardtanh(*args, **kwargs)

class AdaptiveAvgPool2D(symbol.Symbol):
    op_name = opns.ADAPTIVE_AVG_POOL2D

    @property
    def output_size(self) -> typing.Union[int, typing.Tuple[int, int]]:
        assert 'output_size' in self.attrs
        return self.attrs['output_size']

    def __init__(self, X, name=None, output_size:typing.Union[int, typing.Tuple[int, int]]=None, extra_attrs=None):
        assert output_size != None
        super().__init__(X, name=name or N.n(), op_name=opns.ADAPTIVE_AVG_POOL2D, extra_attrs=extra_attrs or {}, **{'output_size': output_size})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['output_size'], **kwargs)

def adaptive_avg_pool2d(*args, **kwargs):
    return AdaptiveAvgPool2D(*args, **kwargs)

class AvgPool2D(symbol.Symbol):
    op_name = opns.AVG_POOL2D

    @property
    def pool_size(self) -> typing.Tuple[int, int]:
        assert 'pool_size' in self.attrs
        return self.attrs['pool_size']
    @property
    def strides(self) -> typing.Tuple[int, int]:
        return self.attrs['strides']
    @property
    def dilation(self) -> typing.Tuple[int, int]:
        return self.attrs['dilation']
    @property
    def padding(self) -> typing.Tuple[int, int, int, int]:
        return self.attrs['padding']
    @property
    def ceil_mode(self) -> bool:
        return self.attrs['ceil_mode']
    @property
    def layout(self) -> str:
        return self.attrs['layout']
    @property
    def count_include_pad(self) -> bool:
        return self.attrs['count_include_pad']

    def __init__(self, X, name=None, pool_size=None, dilation=(1,1), strides=(0,0), padding=(0,0,0,0), ceil_mode=False, layout='NCHW', count_include_pad=True, extra_attrs=None):
        assert pool_size != None
        super().__init__(X, name=name or N.n(), op_name=opns.AVG_POOL2D, extra_attrs=extra_attrs or {}, **{'pool_size':pool_size, 'dilation':dilation, 'strides':strides, 'padding':padding, 'ceil_mode':ceil_mode, 'layout':layout, 'count_include_pad':count_include_pad})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['pool_size', 'dilation', 'strides', 'padding', 'ceil_mode', 'layout', 'count_include_pad'], **kwargs)

def avg_pool2d(*args, **kwargs):
    return AvgPool2D(*args, **kwargs)

class MaxPool2D(symbol.Symbol):
    op_name = opns.MAX_POOL2D

    @property
    def pool_size(self) -> typing.Tuple[int, int]:
        assert 'pool_size' in self.attrs
        return self.attrs['pool_size']
    @property
    def strides(self) -> typing.Tuple[int, int]:
        return self.attrs['strides']
    @property
    def dilation(self) -> typing.Tuple[int, int]:
        return self.attrs['dilation']
    @property
    def padding(self) -> typing.Tuple[int, int, int, int]:
        return self.attrs['padding']
    @property
    def ceil_mode(self) -> bool:
        return self.attrs['ceil_mode']
    @property
    def layout(self) -> str:
        return self.attrs['layout']

    def __init__(self, X, name=None, pool_size=None, dilation=(1,1), strides=(0,0), padding=(0,0,0,0), ceil_mode=False, layout='NCHW', extra_attrs=None):
        assert pool_size != None
        super().__init__(X, name=name or N.n(), op_name=opns.MAX_POOL2D, extra_attrs=extra_attrs or {}, **{'pool_size':pool_size, 'dilation':dilation, 'strides':strides, 'padding':padding, 'ceil_mode':ceil_mode, 'layout':layout})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['pool_size', 'dilation', 'strides', 'padding', 'ceil_mode', 'layout'], **kwargs)

def max_pool2d(*args, **kwargs):
    return MaxPool2D(*args, **kwargs)


class Softmax(symbol.Symbol):
    op_name = opns.SOFTMAX

    @property
    def axis(self) -> typing.Optional[int]:
        return self.attrs['axis']

    def __init__(self, X, name=None, axis=None, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.SOFTMAX, extra_attrs=extra_attrs or {}, **{'axis':axis})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['axis'], **kwargs)

def softmax(*args, **kwargs):
    return Softmax(*args, **kwargs)

class LogSoftmax(symbol.Symbol):
    op_name = opns.LOG_SOFTMAX

    @property
    def axis(self) -> typing.Optional[int]:
        return self.attrs['axis']

    def __init__(self, X, name=None, axis=None, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.LOG_SOFTMAX, extra_attrs=extra_attrs or {}, **{'axis':axis})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['axis'], **kwargs)

def log_softmax(*args, **kwargs):
    return LogSoftmax(*args, **kwargs)


def exp(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.EXP, extra_attrs=extra_attrs or {})

def sigmoid(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.SIGMOID, extra_attrs=extra_attrs or {})

class Sum(symbol.Symbol):
    op_name = opns.SUM

    @property
    def dim(self) -> typing.Optional[typing.Tuple[int, ...]]:
        return self.attrs['dim']

    @property
    def keepdim(self) -> typing.Optional[bool]:
        return self.attrs['keepdim']

    def __init__(self, X, name=None, dim=None, keepdim=None, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.SUM, extra_attrs=extra_attrs or {}, **{'dim': dim, 'keepdim': keepdim})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim', 'keepdim'], **kwargs)

def sum(*args, **kwargs):
    return Sum(*args, **kwargs)


class Mean(symbol.Symbol):
    op_name = opns.MEAN

    @property
    def dim(self) -> typing.Optional[typing.Tuple[int, ...]]:
        return self.attrs['dim']

    @property
    def keepdim(self) -> typing.Optional[bool]:
        return self.attrs['keepdim']

    def __init__(self, X, name=None, dim=None, keepdim=None, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.MEAN, extra_attrs=extra_attrs or {}, **{'dim': dim, 'keepdim': keepdim})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim', 'keepdim'], **kwargs)

def mean(*args, **kwargs):
    return Mean(*args, **kwargs)


class MaxAxis(symbol.Symbol):
    op_name = opns.MAX_AXIS

    @property
    def dim(self) -> typing.Optional[typing.Tuple[int, ...]]:
        return self.attrs['dim']

    @property
    def keepdim(self) -> typing.Optional[bool]:
        return self.attrs['keepdim']

    def __init__(self, X, name=None, dim=None, keepdim=None, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.MAX_AXIS, extra_attrs=extra_attrs or {}, **{'dim': dim, 'keepdim': keepdim})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim', 'keepdim'], **kwargs)

def max_axis(*args, **kwargs):
    return MaxAxis(*args, **kwargs)


def maximum(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.MAXIMUM, extra_attrs=extra_attrs or {})

def minimum(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.MINIMUM, extra_attrs=extra_attrs or {})

def repeat(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.REPEAT, extra_attrs=extra_attrs or {})

class Squeeze(symbol.Symbol):
    op_name = opns.SQUEEZE

    @property
    def dim(self) -> typing.Optional[int]:
        return self.attrs['dim']

    def __init__(self, X, name=None, dim=None, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.SQUEEZE, extra_attrs=extra_attrs or {}, **{'dim': dim})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim'], **kwargs)

def squeeze(*args, **kwargs):
    return Squeeze(*args, **kwargs)

class Flatten(symbol.Symbol):
    op_name = opns.FLATTEN

    @property
    def start_dim(self) -> int:
        return self.attrs['start_dim']

    @property
    def end_dim(self) -> int:
        return self.attrs['end_dim']

    def __init__(self, X, name=None, start_dim=0, end_dim=-1, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.FLATTEN, extra_attrs=extra_attrs or {}, **{'start_dim': start_dim, 'end_dim':end_dim})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['start_dim', 'end_dim'], **kwargs)

def flatten(*args, **kwargs):
    return Flatten(*args, **kwargs)


class Reshape(symbol.Symbol):
    op_name = opns.RESHAPE

    @property
    def newshape(self) -> typing.Tuple[int,...]:
        assert 'newshape' in self.attrs
        return self.attrs['newshape']

    def __init__(self, X, name=None, newshape=None, extra_attrs=None):
        assert newshape != None
        super().__init__(X, name=name or N.n(), op_name=opns.RESHAPE, extra_attrs=extra_attrs or {}, **{'newshape': newshape})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['newshape'], **kwargs)

def reshape(*args, **kwargs):
    return Reshape(*args, **kwargs)

class Concat(symbol.Symbol):
    op_name = opns.CONCAT

    @property
    def axis(self) -> int:
        return self.attrs['axis']

    def __init__(self, X, name=None, axis=None, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.CONCAT, extra_attrs=extra_attrs or {}, **{'axis': axis})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['axis'], **kwargs)

def concat(*args, **kwargs):
    return Concat(*args, **kwargs)

class Split(symbol.Symbol):
    op_name = opns.SPLIT

    @property
    def split_size(self) -> typing.List[int]:
        assert 'split_size' in self.attrs
        return self.attrs['split_size']

    @property
    def dim(self) -> int:
        return self.attrs['dim']

    def __init__(self, X, name=None, split_size=None, dim=0, extra_attrs=None):
        assert split_size != None
        super().__init__(X, name=name or N.n(), op_name=opns.SPLIT, extra_attrs=extra_attrs or {}, **{'split_size': split_size, 'dim': dim})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['split_size', 'dim'], **kwargs)

def split(*args, **kwargs):
    return Split(*args, **kwargs)


class Transpose(symbol.Symbol):
    op_name = opns.TRANSPOSE

    @property
    def dim0(self) -> int:
        assert 'dim0' in self.attrs
        return self.attrs['dim0']

    @property
    def dim1(self) -> int:
        assert 'dim1' in self.attrs
        return self.attrs['dim1']

    def __init__(self, X, name=None, dim0=None, dim1=None, extra_attrs=None):
        assert dim0 != None
        assert dim1 != None
        super().__init__(X, name=name or N.n(), op_name=opns.TRANSPOSE, extra_attrs=extra_attrs or {}, **{'dim0': dim0, 'dim1': dim1})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dim0', 'dim1'], **kwargs)

def transpose(*args, **kwargs):
    return Transpose(*args, **kwargs)


class BroadcastTo(symbol.Symbol):
    op_name = opns.BROADCAST_TO

    @property
    def newshape(self) -> typing.Tuple[int,...]:
        assert 'newshape' in self.attrs
        return self.attrs['newshape']

    def __init__(self, X, name=None, newshape=None, extra_attrs=None):
        assert newshape != None
        super().__init__(X, name=name or N.n(), op_name=opns.BROADCAST_TO, extra_attrs=extra_attrs or {}, **{'newshape': newshape})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['newshape'], **kwargs)

def broadcast_to(*args, **kwargs):
    return BroadcastTo(*args, **kwargs)

class ExpandDims(symbol.Symbol):
    op_name = opns.EXPAND_DIMS

    @property
    def newshape(self) -> typing.Tuple[int,...]:
        assert 'newshape' in self.attrs
        return self.attrs['newshape']

    def __init__(self, X, name=None, newshape=None, extra_attrs=None):
        assert newshape != None
        super().__init__(X, name=name or N.n(), op_name=opns.EXPAND_DIMS, extra_attrs=extra_attrs or {}, **{'newshape': newshape})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['newshape'], **kwargs)

def expand_dims(*args, **kwargs):
    return ExpandDims(*args, **kwargs)

class Tile(symbol.Symbol):
    op_name = opns.TILE

    @property
    def dims(self) -> typing.Tuple[int,...]:
        assert 'dims' in self.attrs
        return self.attrs['dims']

    def __init__(self, X, name=None, dims=None, extra_attrs=None):
        assert dims != None
        super().__init__(X, name=name or N.n(), op_name=opns.TILE, extra_attrs=extra_attrs or {}, **{'dims': dims})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dims'], **kwargs)

def tile(*args, **kwargs):
    return Tile(*args, **kwargs)


def where(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.WHERE, extra_attrs=extra_attrs or {})

def greater(X, Y, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(*[X,Y], name=name or N.n(), op_name=opns.GREATER, extra_attrs=extra_attrs or {})

class NonMaxSuppression(symbol.Symbol):
    op_name = opns.NON_MAX_SUPRESSION

    @property
    def iou_threshold(self) -> float:
        return self.attrs['iou_threshold']
    @property
    def score_threshold(self) -> typing.Optional[float]:
        return self.attrs['score_threshold']

    def __init__(self, X, name=None, iou_threshold=0.5, score_threshold=None, extra_attrs=None):
        super().__init__(X, name=name or N.n(), op_name=opns.NON_MAX_SUPRESSION, extra_attrs=extra_attrs or {}, **{'iou_threshold': iou_threshold,'score_threshold':score_threshold})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['dims'], **kwargs)

def non_max_suppression(*args, **kwargs):
    return NonMaxSuppression(*args, **kwargs)


def ceil(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.CEIL, extra_attrs=extra_attrs or {})

def right_shift(X, Y, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(*[X, Y], name=name or N.n(), op_name=opns.RIGHT_SHIFT, extra_attrs=extra_attrs or {})

class Add(symbol.Symbol):
    op_name = opns.ADD

    @property
    def alpha(self) -> int:
        return self.attrs['alpha']

    def __init__(self, X, Y, name=None, alpha=1, extra_attrs=None):
        super().__init__(*[X, Y], name=name or N.n(), op_name=opns.ADD, extra_attrs=extra_attrs or {}, **{'alpha': alpha})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['alpha'], **kwargs)

def add(*args, **kwargs):
    return Add(*args, **kwargs)

class Sub(symbol.Symbol):
    op_name = opns.SUB

    @property
    def alpha(self) -> int:
        return self.attrs['alpha']

    def __init__(self, X, Y, name=None, alpha=1, extra_attrs=None):
        super().__init__(*[X, Y], name=name or N.n(), op_name=opns.SUB, extra_attrs=extra_attrs or {}, **{'alpha': alpha})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['alpha'], **kwargs)

def sub(*args, **kwargs):
    return Sub(*args, **kwargs)

def mul(X, Y, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(*[X, Y], name=name or N.n(), op_name=opns.MUL, extra_attrs=extra_attrs or {})

def mat_mul(X, Y, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(*[X, Y], name=name or N.n(), op_name=opns.MATMUL, extra_attrs=extra_attrs or {})

class Div(symbol.Symbol):
    op_name = opns.DIV

    @property
    def rounding_mode(self) -> typing.Optional[str]:
        return self.attrs['rounding_mode']

    def __init__(self, X, Y, name=None, rounding_mode=None, extra_attrs=None):
        super().__init__(*[X, Y], name=name or N.n(), op_name=opns.DIV, extra_attrs=extra_attrs or {}, **{'rounding_mode': rounding_mode})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['rounding_mode'], **kwargs)

def div(*args, **kwargs):
    return Div(*args, **kwargs)

def negative(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.NEGATIVE, extra_attrs=extra_attrs or {})

def abs(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.ABS, extra_attrs=extra_attrs or {})

def log(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.LOG, extra_attrs=extra_attrs or {})

def sqrt(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.SQRT, extra_attrs=extra_attrs or {})

def pow(X, Y, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(*[X, Y], name=name or N.n(), op_name=opns.POW, extra_attrs=extra_attrs or {})

def identity(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.IDENTITY, extra_attrs=extra_attrs or {})

class Arange(symbol.Symbol):
    op_name = opns.ARANGE

    @property
    def end(self) -> int:
        assert 'end' in self.attrs
        return self.attrs['end']

    @property
    def start(self) -> int:
        return self.attrs['start']

    @property
    def step(self) -> int:
        return self.attrs['step']

    def __init__(self, name=None, end=None, start=0, step=1, extra_attrs=None):
        assert end != None
        super().__init__(name=name or N.n(), op_name=opns.ARANGE, extra_attrs=extra_attrs or {}, **{'end': end, 'start': start, 'step': step})

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        return _from_dict_attrs(cls, d, ['end', 'start', 'step'], **kwargs)

def arange(*args, **kwargs):
    return Arange(*args, **kwargs)


def zeros_like(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.ZEROS_LIKE, extra_attrs=extra_attrs or {})

def ones_like(X, name=None, extra_attrs=None) -> symbol.Symbol:
    return symbol.Symbol(X, name=name or N.n(), op_name=opns.ONES_LIKE, extra_attrs=extra_attrs or {})


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
_register_op_map(opns.RIGHT_SHIFT)(right_shift)

_register_op_map(opns.ADD)(Add)
_register_op_map(opns.SUB)(Sub)
_register_op_map(opns.MATMUL)(mat_mul)
_register_op_map(opns.DIV)(Div)
_register_op_map(opns.NEGATIVE)(negative)
_register_op_map(opns.ABS)(abs)
_register_op_map(opns.LOG)(log)
_register_op_map(opns.SQRT)(sqrt)
_register_op_map(opns.POW)(pow)
_register_op_map(opns.IDENTITY)(identity)
_register_op_map(opns.ARANGE)(Arange)
_register_op_map(opns.ZEROS_LIKE)(zeros_like)
_register_op_map(opns.ONES_LIKE)(ones_like)


# Add default register Class for MRT OP Not Implemented!
_register_op_map(opns.TUPLE)(extern_opfunc(opns.TUPLE))
_register_op_map(opns.AS_TYPE)(extern_opfunc(opns.AS_TYPE))
_register_op_map(opns.ADV_INDEX)(extern_opfunc(opns.ADV_INDEX))
_register_op_map(opns.CALL_TIR)(extern_opfunc(opns.CALL_TIR))
_register_op_map(opns.CALL_DPS_PACKED)(extern_opfunc(opns.CALL_DPS_PACKED))

_register_op_map(opns.IF)(extern_opfunc(opns.IF))
_register_op_map(opns.ARGWHERE)(extern_opfunc(opns.ARGWHERE))
_register_op_map(opns.REQUANT)(extern_opfunc(opns.REQUANT))
_register_op_map(opns.PCLIP)(extern_opfunc(opns.PCLIP))
_register_op_map(opns.RS_PCLIP)(extern_opfunc(opns.RS_PCLIP))
_register_op_map(opns.LUT)(extern_opfunc(opns.LUT))

_register_op_map(opns.BATCH_FLATTEN)(extern_opfunc(opns.BATCH_FLATTEN))
_register_op_map(opns.STRIDED_SLICE)(extern_opfunc(opns.STRIDED_SLICE))
_register_op_map(opns.SLICE_LIKE)(extern_opfunc(opns.SLICE_LIKE))
_register_op_map(opns.GET_VALID_COUNT)(extern_opfunc(opns.GET_VALID_COUNT))
