import typing

import numpy as np

OpOutputT = typing.Union[list, typing.Any]

OpNumpyT = typing.Union[np.ndarray, list]
ParametersT = typing.Dict[str, OpNumpyT]
AttrsT = typing.Dict[str, typing.Any]

ShapeT = typing.List[int]
""" shape type, list of int, such as [1, 3, 34, 34]. """
DTypeT = str

DataLabelT = typing.Tuple[np.ndarray, typing.Any]
""" a (data, label) representation. """

DefConvertFunc = typing.Callable[[typing.Any], typing.Any]

def to_pydata(value,
                  log_default_type: bool = False,
                  default_convert_func: DefConvertFunc = lambda x: x,
                  support_numpy: bool = True):
    """ PyTorch type to intrinsic py type. """
    # need to pass the kwargs iterately.
    kwargs = {
            "log_default_type": log_default_type,
            "default_convert_func": default_convert_func,
            "support_numpy": support_numpy,
    }
    if isinstance(value, (list, tuple)):
        return [to_pydata(v, **kwargs) for v in value]
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif value is None:
        return value
    elif support_numpy and isinstance(value, np.ndarray):
        return value
    elif log_default_type:
        print(">>> unknown type:", type(value))
    return default_convert_func(value)
