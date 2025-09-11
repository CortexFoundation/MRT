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
    #class CONV2D(Symbol):
    #    strides: typing.Tuple[int, int] = (1,1)
    #    padding: typing.Optional[typing.Tuple[int, int, int, int]] = (0,0,0,0)
    # create mrt op symbol, def func
    print('mrt Conv2D Op Class:', opclass.Conv2D)
    conv2d_b = opclass.MRT_OP_MAP[opns.CONV2D]('conv2d_b',args=[[],[],[]], attrs={'strides':(1,1), 'padding':None})
    assert isinstance(conv2d_b, sx.Symbol), 'not!con2d_b symbol'
    assert isinstance(conv2d_b, opclass.Conv2D), 'not!2 -con2d_b'

    # attrs hint
    assert conv2d_b.args != None
    assert conv2d_b.attrs != None
    assert conv2d_b.strides != None

    print(f'Got {conv2d_b.name} strides: {conv2d_b.strides}')
    print(f'Got {conv2d_b.name} padding: {conv2d_b.padding}')
    print(f'Show {conv2d_b.name} {conv2d_b}')
    return True


# TODO: 
#def test_create_symbol_graph():

if __name__ == "__main__":
    print('MRT_OP_SET as:', opns.MRT_OP_SET)
    assert len(opns.MRT_OP_SET) > 0

    print('MRT_OP_MAP Class as:', opclass.MRT_OP_MAP)
    assert len(opclass.MRT_OP_MAP) > 0
    assert opns.CONV2D in opclass.MRT_OP_MAP

    rltflag = test_create_conv2d_op()
    print("\n" + "="*60 + "\n")
    print('Passed Test!' if rltflag else 'Test Failed!')
    print("\n" + "="*60 + "\n")

