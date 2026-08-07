"""pth → onnx 导出（原 pth2onnx.py，需要 torch + mmdetection 环境）。"""


def export_onnx(checkpoint: str, onnx_path: str, input_shape: tuple = (1, 3, 640, 640)) -> None:
    import torch
    from mmdetection.configs.yolox.yolox_tiny import model as build_model

    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
    model.eval()
    inputs = torch.randn(*input_shape)
    torch.onnx.export(model, inputs, onnx_path, do_constant_folding=False)
    print(f"模型转换成功：{onnx_path}")
