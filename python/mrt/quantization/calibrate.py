from __future__ import annotations

import typing
import numpy as np


from dataclasses import field, InitVar

from mrt.mir import op, opns
from mrt.mir.symbol import *
from mrt.runtime import inference

from mrt.mir.symbol_pass import SymbolTransformer

SamplingFuncT = typing.Callable[
        [typing.Union[OpNumpyT, float]], typing.Any]

class Calibrator(SymbolTransformer):
    @property
    def raw_data(self) -> OpOutputT | None:
        return self.extra_attrs.get("raw_data", None)
    @raw_data.setter
    def raw_data(self, val):
        self.set_extra_attrs(raw_data=val)

    @property
    def data(self) -> typing.List[OpNumpyT]:
        return self.extra_attrs.get("data", [])
    @data.setter
    def data(self, val):
        self.set_extra_attrs(data=val)

    # inherit SymbolParameters __init__
    def __init__(self, *args):
        super().__init__(*args)

    def _rand_data(self,
            enabled: bool = False,
            absmax: float | None = None,
    ) -> np.ndarray:
        assert enabled, "symbol:{} don't have data".format(
                self.name)
        out = np.random.randn(*self.shape)
        out = out.astype(self.dtype)
        if absmax is not None:
            assert absmax > 0
            norm = np.abs(out).max()
            out = out * absmax / norm
        return out

    def __call__(self,
            data: np.ndarray | None = None,
            data_dict: ParametersT = {},
            random_config: typing.Dict[str, typing.Any] = {},
            sampling_func: SamplingFuncT = None,
            **kwargs):
        kwargs.pop("origin", None)

        if self.is_input():
            out = data_dict.get(self.name, data)
            if out is None:
                out = self._rand_data(**random_config)
        elif self.is_param():
            out = self.params[self.name]
        else:
            single_op = op.retrieve_operator(self.graph)
            out = inference.run_single(
                    single_op,
                    [self.from_symbol(a).raw_data for a in self.args],
                    **kwargs)

        assert isinstance(out, (np.ndarray, list)), type(out)
        if isinstance(out, np.ndarray):
            self._assert(out.dtype.name, self.dtype)
            self._assert(out.shape, self.shape)
        else:
            self._assert([o.dtype.name for o in out], self.dtype)
            self._assert([o.shape for o in out], self.shape)

        self.raw_data = out
        if sampling_func is not None:
            out = sampling_func(out)
        self.data.append(out)

    def sampling(self, data):
        if isinstance(data, list):
            return max([self.sampling(d) for d in data]) \
                if data else 0
        return float(np.abs(data).max())

    def __repr__(self, **attrs):
        return super().__repr__(
            data=self.sampling(self.data), **attrs)

    def _assert(self, val, expect):
        if isinstance(val, (list, tuple)):
            assert len(val) == len(expect), (
                    "{} vs. {}").format(val, expect)
            for v, e in zip(val, expect):
                self._assert(v, e)
            return
        assert val == expect, "{} vs. {}".format(val, expect)


class Sampling(SymbolTransformer):
    @property
    def data(self) -> typing.Any:
        return self.extra_attrs.get("data", None)
    @data.setter
    def data(self, val):
        self.set_extra_attrs(data=val)

    # inherit SymbolParameters __init__
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def sampling(cls, np_data: np.ndarray) -> typing.Any:
        raise NotImplementedError()

    def __call__(self, origin: Symbol, **kw):
        print(type(origin), origin)
        if self.is_op(opns.CLIP):
            # TODO: remove clip if threshold is less than a_max
            a_min, a_max = self.parsed.a_min, self.parsed.a_max
            self.data = max(abs(a_min), abs(a_max))
        else:
            self.data = self.sampling(origin.extra_attrs.get("raw_data"))
        return self.graph

class SymmetricMinMaxSampling(Sampling):
    threshold: typing.ClassVar[float] = 1e-5

    # inherit SymbolParameters __init__
    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def sampling(cls, data: typing.List[OpNumpyT]) -> float:
        if isinstance(data, list):
            assert data
            return max([cls.sampling(d) for d in data])
        data = float(np.abs(data).max())
        data = 0 if data < cls.threshold else data
        #  assert data > 0
        return data


