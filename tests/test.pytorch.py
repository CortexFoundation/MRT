import os
from os import path
import sys

ROOT = os.getcwd()
sys.path.insert(0, path.join(ROOT, "python"))

import numpy as np

batch_size = 1
image_shape = (3, 224, 224) # TODO: change the model's input shape
data_shape = (batch_size,) + image_shape

# TODO: set the dataset for target model
# Example: use the torch vision imagenet dataset.
import torch
import torchvision as tv
# data_transform = tv.transforms.Compose([
#     tv.transforms.Resize(256),
#     tv.transforms.CenterCrop(image_shape[1]),
#     tv.transforms.ToTensor(),
#     tv.transforms.Normalize(
#         [0.485,0.456,0.406], [0.229,0.224,0.225])
# ])
data_transform = tv.models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
dataset = tv.datasets.ImageFolder(
        '~/.mxnet/datasets/imagenet/val',
        transform=data_transform)
test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, # set dataset batch load
        )

# use mrt wrapper to uniform api for dataset.
from mrt.dataset.torch import TorchWrapperDataset
ds = TorchWrapperDataset(test_loader)
# prepare data for calibrate pass
data, label = ds.next()
print("test data: ", data.shape, data.flatten()[:10], label)

# model inference context, like cpu, gpu, etc.
config = {
        "device": "cuda:0",
        "target": "" }

# TODO: load the model from torchvision
model_name = "resnet18"                 # passed
#   Iteration: 19 | from_expr: Top1/5: 77.50%,94.69% | sim: Top1/5: 77.50%,94.69% | clip: Top1/5: 77.81%,94.69% | round: Top1/5: 75.62%,94.06% | quantized: Top1/5: 75.31%,94.06% |
# model_name = "mobilenet_v2"             # too much shared data
# model_name = "efficientnet_b2"          # too much shared data
# model_name = "alexnet"                  # passed
#   Iteration: 19 | from_expr: Top1/5: 66.88%,88.44% | sim: Top1/5: 66.88%,88.44% | clip: Top1/5: 67.19%,88.44% | round: Top1/5: 66.56%,89.06% | quantized: Top1/5: 66.56%,89.06% |
# model_name = "densenet121"              # pass
#   Iteration:  19 | from_expr: Top1/5: 83.75%,96.56% | sim: Top1/5: 83.75%,96.56% | clip: Top1/5: 83.75%,96.88% | round: Top1/5: 51.25%,81.25% | quantized: Top1/5: 50.31%,81.25% |
# model_name = "squeezenet1_0"            # pass
#   Iteration:  19 | from_expr: Top1/5: 70.31%,91.56% | sim: Top1/5: 70.31%,91.56% | clip: Top1/5: 70.31%,91.56% | round: Top1/5: 59.06%,85.94% | quantized: Top1/5: 59.38%,85.94% |
# model_name = "vgg11"                    # pass
#   Iteration:  19 | from_expr: Top1/5: 79.06%,95.00% | sim: Top1/5: 79.06%,95.00% | clip: Top1/5: 79.06%,95.00% | round: Top1/5: 77.50%,95.94% | quantized: Top1/5: 77.50%,95.94% |
# model_name = "shufflenet_v2_x0_5"       # pass
#   Iteration:  19 | from_expr: Top1/5: 73.12%,90.00% | sim: Top1/5: 73.12%,90.00% | clip: Top1/5: 73.75%,89.69% | round: Top1/5: 0.94%,4.38% | quantized: Top1/5: 0.94%,4.38% |
model = getattr(tv.models, model_name)(weights="DEFAULT")

model = model.eval()
input_data = torch.randn(data_shape)
ep = torch.export.export(model, (input_data,))
from mrt import frontend as ft
mod, params = ft.model_from_frontend(ep)

#  out = ft.infer(mod, params, data)
#  print(out.shape, out.flatten()[:10])

#  script_module = torch.jit.trace(model, [input_data]).eval()
#  mod, params = relay.frontend.from_pytorch(
#          script_module, [ ("input", data_shape) ])

# MRT Procedure
#  mod: tvm.IRModule = mod
#  func: relay.function.Function = mod["main"]
#  expr: ir.RelayExpr = func.body

from mrt.api import Trace, TraceConfig
from mrt.runtime.analysis import ClassificationOutput
tr = Trace(model_name, "init", mod, params)
# tr = tr.subgraph(onames=["%1"])
tr.bind_dataset(ds, ClassificationOutput).log()

# TODO: test model inference accuracy, uncomment this if don't need.
#  tr.validate_accuracy(
#          tr,
#          max_iter_num=20,
#          **config)
#  sys.exit()

with TraceConfig(
        calibrate_repeats=16,
        force_run_from_trcb="Discretor",
        log_after_all=True,
        #  log_before_tr_or_cbs=[ "calibrate_run_0", ],
        ):
    dis_tr = tr.discrete()

from mrt.quantization import fixed_point as fp
from mrt.common.config import LogConfig
#  with LogConfig(log_vot_cbs=[ "Exporter" ])
with fp.SimConfig:
    sim_tr = dis_tr.exporter(tr_name="sim").log()
with fp.SimIntRequantConfig:
    sim_int_quant_tr = dis_tr.exporter(tr_name="sim-int-quant").log()

#  sim_tr = dis_tr.export("sim").log()
#  sim_clip_tr = dis_tr.export("sim-clip").log()
#  sim_round_tr = dis_tr.export("sim-round").log()
#  sim_quant_tr = dis_tr.export("sim-clip-round").log()
#  fixpt_tr = dis_tr.export("fixpt").log()

tr.validate_accuracy(
        sim_tr,
        sim_int_quant_tr,
        #  sim_clip_tr,
        #  sim_round_tr,
        #  sim_quant_tr,
        max_iter_num=200,
        **config)
sys.exit()
