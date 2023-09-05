# 创建测试时增强的模型
tta_model = dict(
    type="DetTTAModel",
    tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.65), max_per_img=100),
)

# 不同的图像尺寸，用于测试时数据增强
img_scales = [(640, 640), (320, 320), (960, 960)]

# 测试时数据增强管道
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),  # 从文件加载图像
    dict(
        type="TestTimeAug",
        transforms=[
            [
                dict(type="Resize", scale=s, keep_ratio=True)  # 调整大小并保持长宽比
                for s in img_scales
            ],
            [
                # 随机翻转需要放在填充操作之前，否则翻转后的边界框坐标无法正确恢复
                dict(type="RandomFlip", prob=1.0),  # 水平翻转
                dict(type="RandomFlip", prob=0.0),  # 不翻转
            ],
            [
                # 填充为正方形，并使用指定的像素值填充边界
                dict(
                    type="Pad",
                    pad_to_square=True,
                    pad_val=dict(img=(114.0, 114.0, 114.0)),
                ),
            ],
            [dict(type="LoadAnnotations", with_bbox=True)],  # 加载标注框
            [
                # 打包输入数据，便于后续模型的输入和测试
                dict(
                    type="PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "flip",
                        "flip_direction",
                    ),
                )
            ],
        ],
    ),
]
