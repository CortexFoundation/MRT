#!/usr/bin/env python3
"""
Test script to verify dynamic loading of frontends.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mrt.mir.symbol import *
from mrt.mir import op

def test_frontend_loading():
    print("Testing frontend loading...")

    x = op.variable("x", (3, 224), "float")
    w = op.variable("w", (10, 224), "float")
    y = op.nn_dense(x, w)

    # Test PyTorch frontend
    print("\n1. Testing PyTorch frontend:")
    os.environ["FRONTEND"] = "pytorch"
    try:
        from mrt.frontend.api import FRONTEND, model_from_frontend, model_to_frontend
        print(f"   Loaded frontend: {FRONTEND}")
        print(f"   model_from_frontend: {model_from_frontend}")
        print(f"   model_to_frontend: {model_to_frontend}")

        import torch
        fe_model = model_to_frontend(MultiHeadSymbol(main=y), {})
        out = fe_model(torch.randn(3, 224), w = torch.rand(10, 224))
        #  out = fe_model(torch.randn(3, 224))
        #  print(out)
        fe_model = torch.export.export(
                fe_model,
                args=(torch.randn(3, 224),),
                kwargs={
                    "w": torch.randn(10, 224)})
        #  print(fe_model)
        y1 = model_from_frontend(fe_model)

        print("   PyTorch frontend loaded successfully!")
    except Exception as e:
        #  raise e
        print(f"   Error loading PyTorch frontend: {e}")
        return False

    # # Test TVM frontend
    # print("\n2. Testing TVM frontend:")
    # os.environ["FRONTEND"] = "tvm"
    # try:
    #     from mrt.frontend.api import FRONTEND, _DEFAULT_TYPE_INFER, frontend_to_mrt, mrt_to_frontend
    #     print(f"   Loaded frontend: {FRONTEND}")
    #     print(f"   _DEFAULT_TYPE_INFER: {_DEFAULT_TYPE_INFER}")
    #     print(f"   frontend_to_mrt: {frontend_to_mrt}")
    #     print(f"   mrt_to_frontend: {mrt_to_frontend}")
    #     print("   TVM frontend loaded successfully!")
    # except Exception as e:
    #     print(f"   Error loading TVM frontend: {e}")
    #     return False

    # # Test CVM frontend
    # print("\n3. Testing CVM frontend:")
    # os.environ["FRONTEND"] = "cvm"
    # try:
    #     from mrt.frontend.api import FRONTEND, _DEFAULT_TYPE_INFER, frontend_to_mrt, mrt_to_frontend
    #     print(f"   Loaded frontend: {FRONTEND}")
    #     print("   CVM frontend loaded successfully!")
    # except Exception as e:
    #     print(f"   Error loading CVM frontend: {e}")
    #     return False

    # Test invalid frontend
    #  print("\n4. Testing invalid frontend:")
    #  os.environ["FRONTEND"] = "invalid"
    #  try:
    #      from mrt.frontend.api import FRONTEND, _DEFAULT_TYPE_INFER, frontend_to_mrt, mrt_to_frontend
    #      print(f"   ERROR: Should have failed to load invalid frontend!")
    #      return False
    #  except SystemExit:
    #      print("   Correctly exited when trying to load invalid frontend")
    #  except Exception as e:
    #      print(f"   Correctly raised exception when trying to load invalid frontend: {e}")

    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    test_frontend_loading()
