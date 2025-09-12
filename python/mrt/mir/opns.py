""" MRT operator names """
import typing

MRT_OP_SET: typing.Set[str] = set()
def _register_op_list(*op_names: str):
    for op_name in op_names:
        if op_name not in MRT_OP_SET:
            MRT_OP_SET.add(op_name)

VAR = "var"
_register_op_list(VAR)

DROP_OUT = "nn.dropout"
CONV2D = "nn.conv2d"
DENSE = "nn.dense"
BATCH_NORM = "nn.batch_norm"
# BIAS_ADD = "nn.bias_add"
RELU = "nn.relu"
HARDTANH = "nn.hardtanh"
SILU = "nn.silu"
LEAKY_RELU = "nn.leaky_relu"
ADAPTIVE_AVG_POOL2D = "nn.adaptive_avg_pool2d"
AVG_POOL2D = "nn.avg_pool2d"
MAX_POOL2D = "nn.max_pool2d"
_register_op_list(DROP_OUT, CONV2D, DENSE, BATCH_NORM, RELU,
                  HARDTANH, SILU, LEAKY_RELU, ADAPTIVE_AVG_POOL2D,
                   AVG_POOL2D, MAX_POOL2D)

SOFTMAX = "nn.softmax"
LOG_SOFTMAX = "nn.log_softmax"
_register_op_list(SOFTMAX, LOG_SOFTMAX)

EXP = "exp"
SIGMOID = "sigmoid"
_register_op_list(EXP, SIGMOID)

SUM = "sum"
MEAN = "mean"
MAX_AXIS = "max"
MAXIMUM = "maximum"
MINIMUM = "minimum"
_register_op_list(SUM, MEAN, MAX_AXIS, MAXIMUM, MINIMUM)

# =========== NON-CALC ops ===============
TUPLE = "Tuple"
TUPLE_GET_ITEM = "TupleGetItem"
_register_op_list(TUPLE, TUPLE_GET_ITEM)

REPEAT = "repeat"
SQUEEZE = "squeeze"
FLATTEN = "flatten"
BATCH_FLATTEN = "nn.batch_flatten"
RESHAPE = "reshape"
CONCAT = "concatenate"
SPLIT = "split"
TRANSPOSE = "transpose"
BROADCAST_TO = "broadcast_to"
_register_op_list(REPEAT, SQUEEZE, FLATTEN, BATCH_FLATTEN, RESHAPE,
                  CONCAT, SPLIT, TRANSPOSE, BROADCAST_TO, )

EXPAND_DIMS = "expand_dims"
TILE = "tile"
_register_op_list(EXPAND_DIMS, TILE)

WHERE = "where"
GREATER = "greater"
STRIDED_SLICE = "strided_slice"
SLICE_LIKE = "slice_like"
GET_VALID_COUNT = "vision.get_valid_counts"
NON_MAX_SUPRESSION = "vision.non_max_suppression"
_register_op_list(WHERE, GREATER, STRIDED_SLICE, SLICE_LIKE, GET_VALID_COUNT, NON_MAX_SUPRESSION)

# relax clip attrs from a_min/a_max to min/max
CLIP = "clip"
CEIL = "ceil"
RIGHT_SHIFT = "right_shift"
# relax support astype instead of cast
AS_TYPE = "astype"
#  CAST = "cast"
_register_op_list(CLIP, CEIL, RIGHT_SHIFT, AS_TYPE)

ADV_INDEX = "adv_index"
_register_op_list(ADV_INDEX)

CALL_TIR = "call_tir"
CALL_DPS_PACKED = "call_dps_packed"
_register_op_list(CALL_TIR, CALL_DPS_PACKED)

# ======= binary ops =============

ADD = "add"
SUB = "subtract"
MUL = "multiply"
MATMUL = "matmul"
DIV = "divide"
_register_op_list(ADD, SUB, MUL, MATMUL, DIV)

# ======= unary ops ==============

NEGATIVE = "negative"
ABS = "abs"
LOG = "log"
SQRT = "sqrt"
POW = "pow"

PASS = "pass"
_register_op_list(NEGATIVE, ABS, LOG, SQRT, POW, PASS)
# ======= auto generate op =========
ARANGE = "arange"
ZEROS_LIKE = "zeros_like"
ONES_LIKE = "ones_like"
_register_op_list(ARANGE, ZEROS_LIKE, ONES_LIKE)

# ======= control flow op ===========
IF = "if"
ARGWHERE = "argwhere"
_register_op_list(IF, ARGWHERE)

# ======= mrt requant op ==========
REQUANT = "mrt.requant"
PCLIP = "mrt.pclip"
""" precision clip """
RS_PCLIP = "mrt.rs_pclip"
""" right shift precision clip """
LUT = "mrt.lut"
""" look up table, equals adv_index in tvm """
_register_op_list(REQUANT, PCLIP, RS_PCLIP, LUT)


def Opname2Funcname(op_name: str) -> str:
    return op_name.replace('.', '_')
#print('MRT_OP_SET:', MRT_OP_SET)
