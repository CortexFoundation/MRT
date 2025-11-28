"""
Test script for MRT InferPass
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
from mrt.mir import simple_pass

def _get_fasterrcnn_resnet50_fpn_model():
    """Get Fasterrcnn_resnet50_fpn MRT Model"""
    
    # Load pre-trained model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    model.eval()
    
    # Create example input
    example_inputs = torch.randn(1, 3, 224, 224)
    
    # Test inference with original model
    with torch.no_grad():
        original_output = model(example_inputs)
    
    # Convert to MRT
    print("\nConverting Model to MRT...")
    ep = torch.export.export(model, (example_inputs,))
    mrt_graph, mrt_params = pytorch_to_mrt(ep)
    return mrt_graph, mrt_params


def test_InferPass_FuseDivide(mrt_graph, mrt_params):
    symbol = mrt_graph['main']
    #print(symbol)

    print('\n=== Before FuseDivide Pass ===')
    symlist = sx.sym2list(symbol)

    divide_op_cnt = 0
    for sym in symlist:
        divide_op_cnt += 1 if sym.op_name == opns.DIV else 0
    assert divide_op_cnt > 0, f'ori model divide op cnt {divide_op_cnt} == zero!'

    # init FuseDivide Passer and execute visit
    tfs : simple_pass.FuseDividePass = simple_pass.FuseDividePass(symbol, mrt_params)
    symbol_passed = tfs.custom_visits_with_params(tfs.get_run())

    print('\n=== After FuseDivide Pass ===')
    rlts = sx.sym2list(symbol_passed)
    divide_op_cnt_af = 0
    for sym in rlts:
        # print(sym)
        divide_op_cnt_af += 1 if sym.op_name == opns.DIV else 0
    assert divide_op_cnt_af==0, f'passed model divide op cnt {divide_op_cnt_af} != zero!'

    return True


if __name__ == "__main__":
    print("=== Testing InferPass Divide ===")
    mrt_graph, mrt_params = _get_fasterrcnn_resnet50_fpn_model()

    test_id = 0
    passed_cnt = 0
    test_funcs = [test_InferPass_FuseDivide]
    for func_ in test_funcs:
        rltflag = func_(mrt_graph, mrt_params)
        test_id += 1
        passed_cnt += rltflag
        print("\n" + "="*60 + "\n")
        print(f'Passed Test{test_id} Processed({passed_cnt}/{len(test_funcs)}), Passed({passed_cnt}/{test_id})!' if rltflag else f'Test{test_id} Failed! Processed({passed_cnt}/{len(test_funcs)}), Passed({passed_cnt}/{test_id})!')
        print("\n" + "="*60 + "\n")
    print(f'Summary_Passed {passed_cnt}/{len(test_funcs)}')

