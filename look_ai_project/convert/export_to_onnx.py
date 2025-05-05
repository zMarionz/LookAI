# convert/export_to_onnx.py
import torch
from model.cp_vton import CPVTONModel  # presupunem că ai această arhitectură definită
import torchvision.transforms as transforms

model = CPVTONModel()
checkpoint = torch.load("cp_vton.pth", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

dummy_input = torch.randn(1, 6, 256, 192)  # 3 canale user + 3 canale haine
torch.onnx.export(
    model,
    dummy_input,
    "cp_vton.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)
print("Exported model to cp_vton.onnx")