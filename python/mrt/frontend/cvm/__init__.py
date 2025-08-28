"""
CVM frontend for MRT.
"""

# The CVM frontend seems to be more specialized and doesn't have the standard functions
# We'll provide placeholder functions that raise NotImplementedError to indicate
# that CVM is a specialized frontend

def type_infer(symbol):
    raise NotImplementedError("CVM frontend does not support type inference")

def from_frontend(model, *args, **kwargs):
    raise NotImplementedError("CVM frontend does not support conversion from frontend")

def to_frontend(graph, *args, **kwargs):
    raise NotImplementedError("CVM frontend does not support conversion to frontend")

__all__ = ["type_infer", "from_frontend", "to_frontend"]