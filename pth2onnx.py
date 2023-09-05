# import torch
# import torch.onnx
# import os
# from mmdetection.configs.yolox.yolox_tta import tta_model

# def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
#     if not onnx_path.endswith('.onnx'):
#         return 0

#     model = tta_model() #导入模型
#     model.load_state_dict(torch.load(checkpoint)) #初始化权重
#     model.eval()
#     # model.to(device)
    
#     torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
#     print("Exporting .pth model to onnx model has been successful!")

# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES']='0'
#     checkpoint = 'mmdetection\work_dirs\yolox_tiny_8xb8-300e_coco\epoch_300.pth'
#     onnx_path = 'yolox.onnx'
#     input = torch.randn(1, 1, 640, 640)
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#     pth_to_onnx(input, checkpoint, onnx_path)

import torch
 
from mmdetection.configs.yolox.yolox_tiny import model
 
if __name__ == '__main__':
    # model = model(num_classes=4)
    model = model()
    # model.load_state_dict(torch.load("mmdetection\work_dirs\yolox_tiny_8xb8-300e_coco\epoch_300.pth", map_location=torch.device('cpu')))
    inputs = torch.randn(1, 3, 640, 640)
    torch.onnx.export(model, inputs, "./yolox.onnx",  do_constant_folding=False)
    print("模型转换成功!")