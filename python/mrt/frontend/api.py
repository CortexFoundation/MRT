import os
import importlib
import sys
from functools import wraps

from mrt.mir.symbol import *
from mrt.common.types import *

class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
        cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance

class DynamicModule(Singleton):
    def __init__(self):
        self._funcs = {}

    def load_mod(self, frontend):
        try:
            frontend_module = importlib.import_module(f".{FRONTEND}", package="mrt.frontend")
        except ImportError as e:
            print(f"Error: Frontend '{FRONTEND}' cannot be imported: {e}")
            return

        for f in self._funcs:
            if hasattr(frontend_module, f):
                self._funcs[f] = getattr(frontend_module, f)
            else:
                print(f"Error: function '{f}' not found in frontend '{FRONTEND}'")

        return self

    def typedef_mod_function(self, func):
        fname = func.__name__
        self._funcs.setdefault(fname, None)

        @wraps(func)
        def _func_impl(*args, **kwargs):
            assert self._funcs[fname] is not None, f"func:{fname} not registered in mod: {self._funcs.keys()}"
            func(*args, **kwargs)
            return self._funcs[fname](*args, **kwargs)
        return _func_impl


mod = DynamicModule()

@mod.typedef_mod_function
def create_executor(
        symbol: MultiHeadSymbol, params: ParametersT,
        device: str = "cpu",
        target: str = "", # no use in pytorch frontend
        ):
    """ Create Runtime Executor for Model Inference. """
    pass

@mod.typedef_mod_function
def run_executor(
        executor,
        data: typing.Optional[np.ndarray] = None,
        data_dict: ParametersT = {}
        ) -> OpNumpyT:
    """ Apply data to executor. """
    pass

@mod.typedef_mod_function
def infer(
        graph: MultiHeadSymbol,
        params: ParametersT,
        data: typing.Optional[np.ndarray] = None,
        data_dict: ParametersT = {},
        device: str = "cpu",
        **kwargs):
    """ Convinent Method to infer model. """
    pass

@mod.typedef_mod_function
def data_from_frontend(data: typing.Any) -> OpNumpyT:
    """ Convert Frontend Tensor to MRT DType. """
    pass

@mod.typedef_mod_function
def data_to_frontend(data: OpNumpyT):
    """ Convert MRT DType to Frontend Tensor. """
    pass

@mod.typedef_mod_function
def model_from_frontend(
        fe_model,
        func_names: typing.List[str] = [ "main", ]
        ) -> typing.Tuple[MultiHeadSymbol, ParametersT]:
    """ Convert Frontend Graph to MRT Symbol/Params. """
    pass

@mod.typedef_mod_function
def model_to_frontend(graph: MultiHeadSymbol, params: ParametersT,):
    """ Convert MRT Symbol/Params to Frontend Graph. """
    pass

@mod.typedef_mod_function
def type_infer(symbol: Symbol) -> Symbol:
    """ Shape/DType Inference use Frontend API. """


FRONTEND = os.environ.get("FRONTEND", "pytorch")
mod.load_mod(FRONTEND)
