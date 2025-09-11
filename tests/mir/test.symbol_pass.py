"""
Test script for MRT Alexnet FuseDropoutPass.
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
from mrt.mir import symbolpass

def _get_alexnet_model():
    """Get Alexnet MRT Model"""
    
    # Load pre-trained Alexnet
    model = models.alexnet(pretrained=True)
    model.eval()
    
    # Create example input
    example_inputs = torch.randn(1, 3, 224, 224)
    
    # Test inference with original model
    with torch.no_grad():
        original_output = model(example_inputs)
    
    # Convert to MRT
    print("\nConverting Alexnet to MRT...")
    ep = torch.export.export(model, (example_inputs,))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    return mrt_graph, mrt_params

def test_SymbolPass_FuseDropout(mrt_graph, mrt_params):
    symbol = mrt_graph['main']
    #print(symbol)

    print('\n=== Before FuseDropout Pass ===')
    symlist = sx.sym2list(symbol)
    dropout_op_cnt = 0
    for sym in symlist:
        # print(sym)
        dropout_op_cnt += 1 if sym.op_name == opns.DROP_OUT else 0
    assert dropout_op_cnt>0, f'original model dropout op cnt {dropout_op_cnt} == zero!'

    # init FuseDropout Passer and execute visit
    tfs : symbolpass.FuseDropoutPass = symbolpass.FuseDropoutPass(symbol, {})
    #print(getattr(tfs, f"visit_{opns.Opname2Funcname(opns.DROP_OUT)}"))
    symbol_passed = tfs.visit()

    print('\n=== After FuseDropout Pass ===')
    rlts = sx.sym2list(symbol_passed)
    dropout_op_cnt_af = 0
    for sym in rlts:
        # print(sym)
        dropout_op_cnt_af += 1 if sym.op_name == opns.DROP_OUT else 0
    assert dropout_op_cnt_af==0, f'passed model dropout op cnt {dropout_op_cnt_af} != zero!'

    #for sym in symdict:
    #    print(sym, symdict[sym])

    #print('\n=== Back To SymList ===')
    #rltlist = sx.sym2list(symdict[symbol.name])

    return True

if __name__ == "__main__":

    print("=== Testing SymbolPass ===")
    mrt_graph, mrt_params = _get_alexnet_model()

    print("Testing FuseDropoutPass for Model AlexNet")
    rltflag = test_SymbolPass_FuseDropout(mrt_graph, mrt_params)

    print("\n" + "="*60 + "\n")
    print('Passed Test!' if rltflag else 'Test Failed!')
    print("\n" + "="*60 + "\n")

