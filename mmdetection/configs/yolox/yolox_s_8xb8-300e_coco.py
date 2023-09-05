# 导入基础配置文件
_base_ = [
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
    "./yolox_tta.py",
]

img_scale = (640, 640)

# 模型设置
model = dict(
    type="YOLOX",  # 使用YOLOX模型
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,  # 图片pad后的尺寸要是该值的倍数，用于保证特征图的尺寸兼容性
        batch_augments=[
            # 批量增强方式，采用BatchSyncRandomResize进行批量随机尺寸调整
            dict(
                type="BatchSyncRandomResize",
                random_size_range=(480, 800),  # 随机调整尺寸的范围，即最小和最大尺寸
                size_divisor=32,  # 调整后的尺寸要是该值的倍数
                interval=10,  # 调整的时间间隔，即多少个batch调整一次
            )
        ],
    ),
    backbone=dict(
        type="CSPDarknet",  # 使用CSPDarknet作为特征提取网络
        deepen_factor=0.33,  # 深度因子，用于控制特征提取网络的深度
        widen_factor=0.5,  # 宽度因子，用于控制特征提取网络的通道数
        out_indices=(2, 3, 4),  # 指定输出特征图的索引
        use_depthwise=False,  # 是否使用深度可分离卷积
        spp_kernal_sizes=(5, 9, 13),  # SPP层中的核大小
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),  # 归一化层配置
        act_cfg=dict(type="Swish"),  # 激活函数配置
    ),
    neck=dict(
        type="YOLOXPAFPN",  # 使用YOLOXPAFPN作为特征融合网络
        in_channels=[128, 256, 512],  # 输入通道数，来自backbone的特征图通道数
        out_channels=128,  # 输出通道数，即特征融合后的通道数
        num_csp_blocks=1,  # CSP模块的数量
        use_depthwise=False,  # 是否使用深度可分离卷积
        upsample_cfg=dict(scale_factor=2, mode="nearest"),  # 上采样配置
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),  # 归一化层配置
        act_cfg=dict(type="Swish"),  # 激活函数配置
    ),
    bbox_head=dict(
        type="YOLOXHead",  # 使用YOLOXHead作为目标框预测头
        # num_keypoints=2,  # 新增参数：关键点数量
        num_classes=4,  # 类别数量，即预测的目标类别数
        in_channels=128,  # 输入通道数，来自neck的特征图通道数
        feat_channels=128,  # 特征通道数，用于控制目标框预测头的通道数
        stacked_convs=2,  # 堆叠的卷积层数
        strides=(8, 16, 32),  # 特征图的步长
        use_depthwise=False,  # 是否使用深度可分离卷积
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),  # 归一化层配置
        act_cfg=dict(type="Swish"),  # 激活函数配置
        # 损失函数设置
        loss_cls=dict(
            type="CrossEntropyLoss",  # 分类损失函数类型为交叉熵损失
            use_sigmoid=True,  # 是否使用Sigmoid函数，对分类目标使用
            reduction="sum",  # 损失函数的计算方式，"sum"表示将所有样本的损失值相加
            loss_weight=1.0,  # 损失函数的权重，用于平衡各个损失项的重要性
        ),
        loss_bbox=dict(
            type="IoULoss",  # 目标框损失函数类型为IoU损失
            mode="square",  # IoU损失模式为平方模式，即IoU的平方值作为损失
            eps=1e-16,  # 防止分母为零的极小值
            reduction="sum",  # 损失函数的计算方式，"sum"表示将所有样本的损失值相加
            loss_weight=5.0,  # 损失函数的权重，用于平衡各个损失项的重要性
        ),
        loss_obj=dict(
            type="CrossEntropyLoss",  # 目标置信度损失函数类型为交叉熵损失
            use_sigmoid=True,  # 是否使用Sigmoid函数，对目标置信度使用
            reduction="sum",  # 损失函数的计算方式，"sum"表示将所有样本的损失值相加
            loss_weight=1.0,  # 损失函数的权重，用于平衡各个损失项的重要性
        ),
        loss_l1=dict(
            type="L1Loss",  # L1损失函数，用于回归任务
            reduction="sum",  # 损失函数的计算方式，"sum"表示将所有样本的损失值相加
            loss_weight=1.0,  # 损失函数的权重，用于平衡各个损失项的重要性
        ),
        # loss_kp=dict(
        #     type="SmoothL1Loss", beta=1.0, reduction="sum", loss_weight=1.0
        # ),  # 定义关键点损失
    ),
    # 训练配置
    train_cfg=dict(
        assigner=dict(type="SimOTAAssigner", center_radius=2.5)
        # 使用SimOTAAssigner进行目标分配的训练配置，该分配器根据目标的iou和目标中心距离来分配目标
    ),
    # 测试配置
    test_cfg=dict(
        score_thr=0.01,  # 目标分数阈值，低于该阈值的目标将被忽略
        nms=dict(type="nms", iou_threshold=0.65)
        # 非极大值抑制（NMS）配置，用于在测试阶段去除冗余的目标框
    ),
)

# 数据集设置
data_root = "data/coco/"  # 数据集的根目录
dataset_type = "CocoDataset"  # 数据集类型为CocoDataset
classes = (
    "fullcar",
    "cyclist",
    "ped",
    "car",
)  # 数据集的类别列表
# classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


# 使用不同的文件客户端示例
# 方法1：简单设置数据根目录，并让文件I/O模块根据前缀自动推断（不支持LMDB和Memcache）

# data_root = 's3://openmmlab/datasets/detection/coco/'

# 方法2：在3.0.0rc6版本之前使用`backend_args`，`file_client_args`
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))


backend_args = None

# 训练数据处理流程设置
train_pipeline = [
    # dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),  # Mosaic数据增强
    
    # dict(
    #     type="RandomAffine",
    #     scaling_ratio_range=(0.1, 2),
    #     # img_scale is (width, height)
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2),
    # ),  # 随机仿射变换
    # dict(
    #     type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0
    # ),  # MixUp数据增强
    # dict(type="YOLOXHSVRandomAug"),  # YOLOXHSV数据增强
    
    dict(type="RandomFlip", prob=0.5),  # 随机翻转

    # 根据官方实现，多尺度训练不在这里考虑，而在'mmdet/models/detectors/yolox.py'中
    # Resize和Pad在最后15个epoch时关闭Mosaic、RandomAffine和MixUp增强时使用
    dict(type="Resize", scale=img_scale, keep_ratio=True),  # 缩放
    dict(
        type="Pad",
        pad_to_square=True,
        # 如果图像是三通道的，那么填充值需要分别针对每个通道设置。
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),  # 填充
    dict(
        type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False
    ),  # 过滤掉小目标和无目标
    dict(type="PackDetInputs"),  # 打包输入数据
]

# 训练数据集设置
train_dataset = dict(
    # 使用MultiImageMixDataset包装器来支持Mosaic和MixUp增强
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,  # 数据集类型，此处为CocoDataset
        metainfo=dict(classes=classes),  # 数据集的类别信息，包含类别名称
        data_root=data_root,  # 数据集根目录
        ann_file="annotations/instances_train2017.json",  # 训练集的标注文件路径
        data_prefix=dict(img="train2017/"),  # 图像文件路径前缀
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=backend_args),  # 从文件加载图像
            dict(type="LoadAnnotations", with_bbox=True),  # 从文件加载标注框
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 过滤掉空标注框和小尺寸标注框
        backend_args=backend_args,  # 文件加载器参数
    ),
    pipeline=train_pipeline,  # 训练数据的预处理管道
)

# 测试数据的预处理管道
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),  # 从文件加载图像
    dict(type="Resize", scale=img_scale, keep_ratio=True),  # 缩放图像，保持长宽比
    dict(
        type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))
    ),  # 将图像填充为正方形
    dict(type="LoadAnnotations", with_bbox=True),  # 从文件加载标注框
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),  # 打包输入数据，用于推理时使用
]

# 训练数据加载器设置
train_dataloader = dict(
    batch_size=1,  # 每个批次的样本数量
    num_workers=4,  # 数据加载器的工作进程数
    persistent_workers=True,  # 是否持久化工作进程
    sampler=dict(type="DefaultSampler", shuffle=True),  # 默认采样器，用于随机采样
    dataset=train_dataset,  # 训练数据集
)

# 验证数据加载器设置
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),  # 默认采样器，用于顺序采样
    dataset=dict(
        type=dataset_type,  # 数据集类型，此处为CocoDataset
        metainfo=dict(classes=classes),  # 数据集的类别信息，包含类别名称
        data_root=data_root,  # 数据集根目录
        ann_file="annotations/instances_val2017.json",  # 验证集的标注文件路径
        data_prefix=dict(img="val2017/"),  # 图像文件路径前缀
        test_mode=True,  # 设置为True，表明是测试模式
        pipeline=test_pipeline,  # 测试数据的预处理管道
        backend_args=backend_args,  # 文件加载器参数
    ),
)

test_dataloader = val_dataloader  # 测试数据加载器与验证数据加载器相同

val_evaluator = dict(
    type="CocoMetric",  # 使用CocoMetric作为验证评估器
    ann_file=data_root + "annotations/instances_val2017.json",  # 验证数据集的标注文件
    metric="bbox",  # 使用bbox指标进行评估
    backend_args=backend_args,
)  # 文件I/O相关参数
test_evaluator = val_evaluator  # 测试评估器与验证评估器相同

# 训练设置
max_epochs = 300  # 最大训练轮次
num_last_epochs = 15  # 最后几个epoch
interval = 1  # 每隔几个epoch进行一次验证

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)  # 训练配置

# 优化器设置
# 默认8个GPU
base_lr = 0.0001  # 基础学习率
optim_wrapper = dict(
    type="OptimWrapper",  # 优化器包装器类型
    optimizer=dict(
        type="SGD",  # 使用SGD优化器
        lr=base_lr,  # 学习率
        momentum=0.9,  # 动量
        weight_decay=5e-4,  # 权重衰减
        nesterov=True,
    ),  # 使用Nesterov加速梯度下降
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)  # 参数权重设置

# 学习率设置
param_scheduler = [
    dict(
        # 使用二次函数进行前5个epoch的warm-up，并且学习率根据迭代次数进行更新
        # TODO: 修复get函数中默认范围的问题
        type="mmdet.QuadraticWarmupLR",
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        # 使用余弦衰减学习率，从第5个epoch衰减到第285个epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,  # 最小学习率，基于初始学习率的倍数
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        # 最后15个epoch保持固定学习率
        type="ConstantLR",
        by_epoch=True,
        factor=1,  # 学习率缩放因子，此处不改变学习率
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    ),
]

# 默认钩子设置
default_hooks = dict(
    checkpoint=dict(
        interval=interval,  # 每隔interval个epoch保存一次checkpoint
        max_keep_ckpts=3,  # 最多保留最近的3个checkpoint
    )
)

# 自定义钩子设置
custom_hooks = [
    dict(
        type="YOLOXModeSwitchHook",  # YOLOX模式切换钩子，用于在训练的最后几个epoch切换YOLOX模式
        num_last_epochs=num_last_epochs,
        priority=48,  # 钩子的优先级，数值越大，优先级越高
    ),
    dict(type="SyncNormHook", priority=48),  # 同步归一化钩子，用于对BN层的权重进行同步归一化
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",  # 使用指数滑动平均（EMA）更新模型参数
        momentum=0.0001,  # EMA的动量值，控制历史参数的权重大小
        update_buffers=True,  # 是否更新模型的EMA缓冲区
        priority=49,  # 钩子的优先级，数值越大，优先级越高
    ),
]


# 注意: `auto_scale_lr` 是用于自动缩放学习率的参数，用户不应更改其值。
# 基础批次大小 = (8个GPU) x (每个GPU 8个样本)
auto_scale_lr = dict(base_batch_size=64)
