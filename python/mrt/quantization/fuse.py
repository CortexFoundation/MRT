from dataclasses import InitVar
from collections import namedtuple

import numpy as np

from mrt.mir import opclass, optype
from mrt.mir.opns import *
from mrt.mir.symbol import *
from mrt.mir.attrs import *

from mrt.runtime import inference
from mrt.common.utils import N, product

from mrt.mir.symbol_pass import SymbolTransformer


class FuseDropout(SymbolTransformer):
    #out = filter_operators(DROP_OUT)(__call__)
    # def out():
    @filter_operators(DROP_OUT)
    def __call__(self, **kwargs):
        return self.args[0]

class FuseIdentity(SymbolTransformer):
    @filter_operators(IDENTITY)
    def __call__(self, **kwargs):
        return self.args[0]

class FuseConstant(SymbolTransformer):
    threshold: typing.ClassVar[float] = 1e-5

    def np_is_zero(self, data) -> float:
        return np.abs(data).max() < self.threshold

    def __call__(self: SymbolTransformer, **kw):
        if self.is_operator() and all([self.from_symbol(c).is_param() for c in self.args]):
            data = inference.run_single(
                    self.graph, [self.from_symbol(a).numpy() for a in self.args])
            return self.as_parameter(data)
        elif self.is_op(ADD, SUB): # , BIAS_ADD):
            strips = []
            for arg in self.args:
                if self.from_symbol(arg).is_param() and self.np_is_zero(self.from_symbol(arg).numpy()):
                #  if arg.is_param() and np.abs(arg.numpy()).max() == 0:
                    strips.append(arg)
            args = [a for a in self.args if a not in strips]
            if len(args) == 1:
                return args[0]
        elif self.is_op(SLICE_LIKE):
            if not self.from_symbol(self.args[0]).is_param():
                return
            a, b = self.args
            arg1 = np.zeros(b.shape, b.dtype)
            data = inference.run_single(
                self.graph, [self.from_symbol(a).numpy(), np.zeros(b.shape, b.dtype)])
            return self.as_parameter(data)
        elif self.is_op(REQUANT):
            if self.parsed.rescale == 1:
                return self.args[0]
        elif self.is_op(ZEROS_LIKE, ONES_LIKE):
            data = inference.run_single(self.graph, [])
            return self.as_parameter(data)


class FuseBatchNorm(SymbolTransformer):
    @filter_operators(BATCH_NORM)
    def __call__(self, **kw):
        X, gamma, beta, mean, var = self.args
        X = self.from_symbol(X)
        parsed: BatchNormAttrs = self.parsed

        gamma, beta = self.from_symbol(gamma).numpy(), self.from_symbol(beta).numpy()
        mean, var = self.from_symbol(mean).numpy(), self.from_symbol(var).numpy()
        #  print(gamma.shape, beta.shape, mean.shape, var.shape)

        assert parsed.axis == 1
        beta = beta if parsed.center else 0
        gamma = gamma if parsed.scale else 1

        # (X - mean) / sqrt(var + epsilon) * gamma + beta
        gamma = gamma / np.sqrt(var + parsed.epsilon)
        # (X - mean) * gamma + beta
        # X * gamma + (beta - mean * gamma)
        bias: np.ndarray = (beta - mean * gamma)
        #  print(np.abs(gamma).max(), np.abs(bias).max())
        # X * gamma + bias

        if X.is_op(CONV2D):
            A, W = X.args
            conv_parsed: Conv2DAttrs = X.parsed

            # assert conv_parsed.kernel_layout == "OIHW"
            K = gamma.shape[0]
            assert W.shape[0] == K

            # (A * W) * gamma + bias
            # A * (W * gamma) + bias
            W_data = self.from_symbol(W).numpy() * gamma.reshape(K, 1, 1, 1)
            W_sym = self.from_symbol(W).from_np_data(W_data)
            out = optype.infer_single(opclass.conv2d(A, W_sym, **X.attrs))
        elif X.is_op(DENSE):
            A, W = X.args
            dense_parsed: DenseAttrs = X.parsed

            # (A * W) * gamma + bias
            # A * (W * gamma) + bias
            W_data = self.from_symbol(W).numpy() * gamma.reshape(K, 1)
            W_sym = self.from_symbol(W).from_np_data(W_data)
            out = optype.infer_single(opclass.dense(A, W_sym, **X.attrs))
        else:
            reshp = [s if i == parsed.axis else 1 \
                    for i, s in enumerate(X.shape)]
            W = X.from_np_data(gamma.reshape(reshp))
            out = optype.infer_single(opclass.mul(X.graph, W))

        bias = bias.reshape([s if i == parsed.axis else 1 \
                for i, s in enumerate(out.shape)])
        B = self.from_symbol(out.like(self.graph)).from_np_data(bias)
        out = opclass.add(out, B)
        out = optype.infer_single(out)
        return out.like(self.graph)

class FuseTupleGetItem(SymbolTransformer):
    @filter_operators(TUPLE_GET_ITEM)
    def __call__(self, **kw):
        X: Symbol = self.args[0]
        if self.from_symbol(X).is_op(BATCH_NORM, DROP_OUT):
            return X
        #  assert X.is_op(BATCH_NORM, DROP_OUT), X.name
        #  assert self.parsed.index == 0
        #  return X

class FuseAvgPool2D(SymbolTransformer):
    def __call__(self, **kw):
        out = self._fuse_adaptive_avg_pool2d()
        out = out or self._fuse_avg_pool2d()
        return out

    @filter_operators(AVG_POOL2D)
    def _fuse_avg_pool2d(self):
        X: Symbol = self.args[0]
        parsed: AvgPool2DAttrs = self.parsed
        assert parsed.layout == "NCHW"
        # TODO: ignore for unstrict mode
        assert parsed.count_include_pad == True
        attrs = {
            "kernel_size": parsed.pool_size,
            "strides": parsed.strides,
            "padding": parsed.padding,
            "dilation": parsed.dilation,
            "data_layout": parsed.layout,
            "groups": X.shape[1],
            "channels": X.shape[1],
            }
        W_shape = (X.shape[1], 1, *parsed.pool_size)
        W = self.from_symbol(X).from_np_data(np.full(
            W_shape, 1 / product(parsed.pool_size)))
        out = optype.infer_single(opclass.conv2d(X, W, **attrs))
        return out.like(self.graph)


    @filter_operators(ADAPTIVE_AVG_POOL2D)
    def _fuse_adaptive_avg_pool2d(self):
        X: Symbol = self.args[0]
        parsed: AdaptiveAvgPool2DAttrs = self.parsed
        assert parsed.layout == "NCHW"
        ins = X.shape[2:]
        ous = parsed.output_size or ins
        if not isinstance(ous, (list, tuple)):
            ous = (ous, ous)
        parsed.output_size = ous

        assert len(X.shape) == 4
        if all([s == 1 for s in parsed.output_size]):
            scale = np.array(1 / np.prod(X.shape[-2:]))
            out = optype.infer_single(opclass.sum(X, dim=list(range(4))[-2:], keepdim=True))
            scale = self.from_np_data(scale.astype(X.dtype))
            out = optype.infer_single(opclass.mul(out, scale))
            return out.like(self.graph)
        elif ous[0] > ins[0] or ous[1] > ins[1]:
            assert all([s == 1 for s in ins])
            out = optype.infer_single(opclass.repeat(X, repeats=ous[0], axis=-2))
            out = optype.infer_single(opclass.repeat(out, repeats=ous[1], axis=-1))
            return out.like(self.graph)

        # calculate the attributes refers to:
        # https://stackoverflow.com/questions/53841509/how-does-adaptive-pooling-in-pytorch-work
        strides = [i // o for i, o in zip(ins, ous)]
        kernel = [i-(o-1)*s for i, o, s in zip(ins, ous, strides)]
        attrs = {
            "kernel_size": kernel,
            "strides": strides,
            "padding": (0, 0),
            "dilation": (1, 1),
            "data_layout": parsed.layout,
            "groups": X.shape[1],
            "channels": X.shape[1],
        }
        W_shape = (X.shape[1], 1, *kernel)
        W = self.from_symbol(X).from_np_data(np.full(W_shape, 1 / product(kernel)))
        out = optype.infer_single(opclass.conv2d(X, W, **attrs))
        return out.like(self.graph)

class FuseNaiveSoftmax(SymbolTransformer):
    def __call__(self, **kw):
        return self.graph # not fuse pass

        if self.is_op(SOFTMAX, LOG_SOFTMAX):
            return self.args[0]
        assert self.is_variable() or not self.from_symbol(self.args[0]).is_op(SOFTMAX, LOG_SOFTMAX)
        return self.graph

class FuseMean(SymbolTransformer):
    @filter_operators(MEAN)
    def __call__(self, **kw):
        X: Symbol = self.args[0]
        #  max_axis = len(X.shape)
        #  axis = X.attrs.get("axis", None)
        #  axis = axis or [i for i in range(max_axis)]
        #  axis = [a if a >= 0 else a + max_axis for a in axis]
        #  assert all([a >= 0 and a < max_axis for a in axis])
        #  if exclude:
        #      axis = [a for a in range(max_axis) if a not in axis]
        #  axis_len = product([X.shape[a] for a in axis])

        out = optype.infer_single(opclass.sum(X, **self.attrs))
        scale = self.from_np_data(np.array(
            1. * product(out.shape) / product(X.shape)))
        out = optype.infer_single(opclass.mul(out, scale))
        return out.like(self.graph)

class FuseLeakyReLU(SymbolTransformer):
    @filter_operators(LEAKY_RELU)
    def __call__(self, **kw):
        """ Customized rewrite pass Introduction.

            LeakyReLU can be equivalently transformed.

            .. math::
                LeakyReLU(X) = relu(X) - slope*relu(-X)
        """
        alpha = self.from_const_data(self.parsed.alpha)
        X: Symbol = self.args[0]
        out = optype.infer_single(opclass.negative(X))
        out = optype.infer_single(opclass.relu(out))
        out = optype.infer_single(opclass.mul(alpha, out))
        out = optype.infer_single(opclass.sub(optype.infer_single(opclass.relu(X)), out))
        return out.like(self.graph)


class FuseDivide(SymbolTransformer):
    @filter_operators(DIV)
    def __call__(self, **kw):
        """ Transform div to mul if possible. """
        A: Symbol = self.args[0]
        B: Symbol = self.args[1]
        assert self.from_symbol(B).is_param(), B
        B = self.from_symbol(B).from_np_data(1. / self.from_symbol(B).numpy())
        out = optype.infer_single(opclass.mul(A, B))
        return out.like(self.graph)

# move to fuse constant
#  class FuseNaiveMathmatic(SymbolTransformer):
#      def __call__(self):
#          if self.is_op(BIAS_ADD):
#              X, B = self.args
#              if B.is_param() and np.abs(B.numpy()).max() == 0:
#                  return X




