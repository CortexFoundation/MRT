import typing

import numpy as np

try:
    import tvm
    OpOutputT = typing.Union[tvm.nd.NDArray, list]
except Exception:
    print("TVM data type failed.")

OpNumpyT = typing.Union[np.ndarray, list]
ParametersT = typing.Dict[str, OpNumpyT]
AttrsT = typing.Dict[str, typing.Any]

ShapeT = typing.List[int]
""" shape type, list of int, such as [1, 3, 34, 34]. """
DTypeT = str

DataLabelT = typing.Tuple[np.ndarray, typing.Any]
""" a (data, label) representation. """
