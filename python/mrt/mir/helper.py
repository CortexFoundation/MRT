import typing

from .symbol import *
from mrt.common.types import ParametersT

from . import op, optype

def set_input_shape(
        symbol: Symbol, params: ParametersT,
        shape = None, shape_dict = {}) -> Symbol:
    # print(shape, shape_dict)
    def _set_shape(sym: Symbol):
        if not op.is_input(sym, params):
            return
        tshape = shape_dict.get(sym.name, shape)
        assert tshape is not None, sym
        if list(sym.shape) != list(tshape):
            print("change {}'s shape from {} into {}".format(
                sym.name, sym.shape, tshape))
        sym.shape = tshape
        return sym
    out = transform(symbol, _set_shape)

    return optype.infer(out)
    # out = op.infer_type(out)
    # out = op.graph_like(out, symbol)
    # return out

def format_symbol(
        graph: Symbol, params: ParametersT,
        name: str = "", # hint name for print header
        prefix: int = 0, # prefix layers to print
        suffix: int = 0, # suffix layers to print
        short: bool = False, # 5 prefix and suffix by short
        till_layer: int = 0, # no current layer to print
        selects: typing.List[str] = [], # print ops or names
        ) -> str:
    header = "{f} {s} View {f}\n".format(f="=" * 25, s=name)

    info = {
        "ops": 0, "params": 0,
        "op_names": set(),
        "curr_layer": 0, "total_layers": 0,
        "msg": header,
    }
    def _calc(sym: Symbol):
        info["total_layers"] += 1
        info["op_names"].add(sym.op_name)
        if op.is_param(sym, params):
            info["params"] += np.prod(sym.shape or (0))
        info["ops"] += op.is_operator(sym)

    visit(graph, _calc)

    if short:
        prefix = prefix or 5
        suffix = suffix or 5

    till_layer = till_layer or info["total_layers"]
    prefix  = prefix or till_layer
    suffix = suffix or till_layer
    suffix = till_layer - suffix

    selects = selects or info["op_names"]
    def _print(sym: Symbol):
        layer = info["curr_layer"]
        info["curr_layer"] += 1

        if suffix > prefix and layer == suffix:
            info["msg"] += "\t.....\n\t{{ skip {} layers }}\n".format(
                        suffix - prefix)

        passed = layer < till_layer
        passed = passed and (layer < prefix or layer >= suffix)
        selected = sym.name in selects or sym.op_name in selects
        passed = passed and selected
        if passed:
            info["msg"] += str(sym) + "\n"

    visit(graph, _print)

    msg = info["msg"]
    msg += "_" * len(header) + "\n"
    msg += "Layers: {} | Operators: {} | Parameters: {}\n".format(
        info["total_layers"], info["ops"], int(info["params"]))
    msg += f"Operator Names: {', '.join(info['op_names'])}\n"
    msg += "=" * len(header) + "\n"
    return msg

def format_print(*args, **kwargs):
    print(format_symbol(*args, **kwargs))



