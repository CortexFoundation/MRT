import os
import importlib
import sys

from mrt.mir.optype import _DEFAULT_TYPE_INFER

FRONTEND = os.environ.get("FRONTEND", "pytorch")

# Dynamically load the frontend module
try:
    frontend_module = importlib.import_module(f".{FRONTEND}", package="mrt.frontend")
except ImportError as e:
    print(f"Error: Frontend '{FRONTEND}' is not supported or cannot be imported: {e}")
    sys.exit(1)

# Register default type infer functions
if hasattr(frontend_module, "type_infer"):
    _DEFAULT_TYPE_INFER = frontend_module.type_infer
else:
    print(f"Error: Required function 'type_infer' not found in frontend '{FRONTEND}'")
    sys.exit(1)

# Try to get frontend_to_mrt function
frontend_to_mrt = None
if hasattr(frontend_module, "from_frontend"):
    frontend_to_mrt = frontend_module.from_frontend
elif hasattr(frontend_module, "pytorch_to_mrt"):
    frontend_to_mrt = frontend_module.pytorch_to_mrt
elif hasattr(frontend_module, "expr2symbol"):
    frontend_to_mrt = frontend_module.expr2symbol
else:
    print(f"Error: Required function 'frontend_to_mrt' not found in frontend '{FRONTEND}'")
    sys.exit(1)

# Try to get mrt_to_frontend function
mrt_to_frontend = None
if hasattr(frontend_module, "to_frontend"):
    mrt_to_frontend = frontend_module.to_frontend
elif hasattr(frontend_module, "mrt_to_pytorch"):
    mrt_to_frontend = frontend_module.mrt_to_pytorch
elif hasattr(frontend_module, "symbol2expr"):
    mrt_to_frontend = frontend_module.symbol2expr
else:
    print(f"Error: Required function 'mrt_to_frontend' not found in frontend '{FRONTEND}'")
    sys.exit(1)
