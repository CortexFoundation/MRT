"""
Test script for Alexnet PyTorch to MRT conversion.
"""

from os import path
import sys, os

ROOT = path.dirname(path.dirname(path.dirname(
    path.realpath(__file__))))
sys.path.insert(0, path.join(ROOT, "python"))

import torch
import torchvision.models as models
import numpy as np
from collections import namedtuple

from mrt.frontend.pytorch import pytorch_to_mrt, mrt_to_pytorch, type_infer
from mrt.frontend.pytorch import vm
from mrt.mir import helper, symbol as sx
from mrt.mir import opns
from mrt.mir import opclass


def test_create_conv2d_op():

    X = opclass.Variable(name="x", shape=(1, 3, 224, 224,), dtype="float")
    W = opclass.Variable(name="w", shape=(32, 3, 10, 10,), dtype="float")
    assert [shp for shp in X.shape] == [shp for shp in (1, 3, 224, 224)], f'Wrong X shape {X.shape}'
    assert X.dtype == "float", f'Wrong X dtype {X.dtype}'

    # Symbol Init using opclass OP
    conv2d_a = opclass.Conv2D(X, W, name='conv2d_a', strides=(2,2))
    assert isinstance(conv2d_a, sx.Symbol), 'conv2d_a isnot a symbol'
    assert isinstance(conv2d_a, opclass.Conv2D), 'conv2d_a isnot a Conv2D'

    # attrs hint
    assert conv2d_a.args != None
    assert conv2d_a.attrs != None
    assert conv2d_a.strides != None

    print(f'Got {conv2d_a.name} strides: {conv2d_a.strides}')
    print(f'Got {conv2d_a.name} padding: {conv2d_a.padding}')
    print(f'Show {conv2d_a.name} {conv2d_a}')

    # test Conv2D clone mode
    conv2d_b = conv2d_a.copy()
    assert isinstance(conv2d_b, sx.Symbol), 'conv2d_b isnot a symbol'
    assert isinstance(conv2d_b, opclass.Conv2D), 'conv2d_b isnot a Conv2D'

    assert conv2d_b.attrs == conv2d_a.attrs, f'a: {conv2d_b.attrs} != b: {conv2d_a.attrs}'

    # test Dict to Find Class and Init
    conv2d_c = opclass.MRT_OP_MAP[opns.CONV2D](X, W, strides=(2,2))
    assert isinstance(conv2d_c, opclass.Conv2D), 'conv2d_c isnot a Conv2D'

    # test Variable clone mode
    X1 = X.copy()
    assert X1.shape == X.shape
    assert X1.dtype == X.dtype

    # test: Symbol Compatible Mode
    args = [X1, W]
    attrs = {'strides':(3,3)}

    # Symbol Compatible Init
    conv2d_d = opclass.Conv2D(*args, name='conv2d_d', **attrs)
    conv2d_e = opclass.Conv2D(*args, **attrs)
    assert isinstance(conv2d_d, opclass.Conv2D), 'conv2d_d isnot a Conv2D'
    assert isinstance(conv2d_e, opclass.Conv2D), 'conv2d_e isnot a Conv2D'

    return True


def test_create_symbol_graph():
    X0 = opclass.Variable(name="x", shape=(1, 3, 224, 224,), dtype="float")
    W0 = opclass.Variable(name="w", shape=(32, 3, 10, 10,), dtype="float")
    conv2d_a = opclass.Conv2D(X0, W0, name='conv2d_a', strides=(1,1))

    W1 = opclass.Variable(shape=(16, 3, 12, 12,), dtype="float")
    conv2d_b = opclass.Conv2D(conv2d_a, W1, name='conv2d_b', strides=(1,1))
    symlist = sx.sym2list(conv2d_b)

    assert symlist[0] == X0
    assert symlist[1] == W0

    for id_ in range(len(symlist)):
        print(id_, symlist[id_])

    return True


def test_create_batch_norm_op():
    X = opclass.Variable(name="x", shape=(1, 32, 128, 128,), dtype="float")
    Gamma = opclass.Variable(name="gamma", shape=(32,), dtype="float")
    Beta = opclass.Variable(name="beta", shape=(32,), dtype="float")
    Mean = opclass.Variable(name="mean", shape=(32,), dtype="float")
    Var = opclass.Variable(name="var", shape=(32,), dtype="float")
    batch_norm_a = opclass.BatchNorm(X, Gamma, Beta, Mean, Var, axis=1, epsilon=1e-4)

    # attrs hint
    assert batch_norm_a.args != None
    assert batch_norm_a.attrs != None
    assert batch_norm_a.axis != 0

    # test clone mode
    batch_norm_b = batch_norm_a.copy()
    assert isinstance(batch_norm_b , opclass.BatchNorm)

    assert batch_norm_a.attrs == batch_norm_b.attrs, f'a: {batch_norm_a.attrs} != b: {batch_norm_b.attrs}'
    assert len(batch_norm_a.args) == len(batch_norm_b.args), f'a: {len(batch_norm_a.args)} != b: {len(batch_norm_b.args)}'

    return True


if __name__ == "__main__":
    print('MRT_OP_SET as:', opclass.MRT_OP_MAP.keys())
    assert len(opclass.MRT_OP_MAP.keys()) > 0

    assert opns.CONV2D in opclass.MRT_OP_MAP
    print('MRT_OP_MAP Conv2D Class as:', opclass.MRT_OP_MAP[opns.CONV2D])

    test_id = 0
    for func_ in [test_create_conv2d_op, test_create_symbol_graph, test_create_batch_norm_op]:
        rltflag = func_()
        test_id += 1
        print("\n" + "="*60 + "\n")
        print(f'Passed Test{test_id}!' if rltflag else f'Test{test_id} Failed!')
        print("\n" + "="*60 + "\n")

