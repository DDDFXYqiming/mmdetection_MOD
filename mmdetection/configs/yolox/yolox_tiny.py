# 导入基础配置文件
_base_ = "./yolox_s_8xb8-300e_coco.py"

# 模型设置
model = dict(
    # 数据预处理器设置，用于在数据加载时进行批量增强
    data_preprocessor=dict(
        batch_augments=[
            # 使用BatchSyncRandomResize进行批量随机尺寸调整
            dict(
                type="BatchSyncRandomResize",
                random_size_range=(320, 640),  # 随机调整尺寸的范围，即最小和最大尺寸
                size_divisor=32,  # 调整后的尺寸要是该值的倍数
                interval=1,  # 调整的时间间隔，即多少个batch调整一次
            )
        ]
    ),
    # 特征提取网络（backbone）设置
    backbone=dict(
        deepen_factor=0.33,  # 深度因子，用于控制特征提取网络的深度
        widen_factor=0.375,  # 宽度因子，用于控制特征提取网络的通道数
    ),
    # 特征融合网络（neck）设置
    neck=dict(
        in_channels=[96, 192, 384],  # 输入通道数，来自backbone的特征图通道数
        out_channels=96,  # 输出通道数，即特征融合后的通道数
    ),
    # 目标框预测头（bbox_head）设置
    bbox_head=dict(
        in_channels=96, feat_channels=96  # 输入通道数，来自neck的特征图通道数  # 特征通道数，用于控制目标框预测头的通道数
    ),
)


img_scale = (640, 640)

# 训练数据处理管道
train_pipeline = [
    dict(
        type="Mosaic", img_scale=img_scale, pad_val=114.0
    ),  # 使用Mosaic进行数据增强，将多张图片拼接成一张，增加样本的多样性
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.5, 1.5),  # 随机仿射变换的缩放范围
        border=(-img_scale[0] // 2, -img_scale[1] // 2),  # 随机仿射变换的边界
    ),  # 随机仿射变换，用于增加样本的多样性
    dict(type="YOLOXHSVRandomAug"),  # YOLOXHSV随机增强，改变图片的颜色空间，增加样本的多样性
    dict(type="RandomFlip", prob=0.5),  # 随机翻转，以一定概率对图片进行水平翻转，增加样本的多样性
    # 在最后的15个epochs时，关闭Mosaic和RandomAffine，只进行Resize和Pad操作
    dict(type="Resize", scale=img_scale, keep_ratio=True),  # 调整大小并保持长宽比，将图片缩放到指定大小
    dict(
        type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
    ),  # 填充为正方形，并使用指定的像素值填充边界
    dict(
        type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False
    ),  # 过滤小目标和空标签，去除尺寸过小的标注框和没有标注框的样本
    dict(type="PackDetInputs"),  # 将处理后的数据打包成批次，便于后续模型的输入
]


# 测试数据处理管道
test_pipeline = [
    # dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),  # 从文件中加载图像
    dict(type="Resize", scale=(416, 416), keep_ratio=True),  # 调整大小并保持长宽比
    dict(
        type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
    ),  # 填充为正方形，并使用指定的像素值填充边界
    dict(type="LoadAnnotations", with_bbox=True),  # 从文件中加载目标标注信息
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),  # 将处理后的数据打包成批次，并保存一些元数据信息
]

# 训练数据加载器设置
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

# 验证数据加载器设置，使用和测试数据处理管道相同的设置
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# 测试数据加载器设置，使用和测试数据处理管道相同的设置
test_dataloader = val_dataloader
