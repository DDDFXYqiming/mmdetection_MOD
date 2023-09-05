# 训练计划设置：1x 训练循环，总共训练 12 个 epochs，每 1 个 epoch 进行一次验证
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

# 验证计划设置：每个 epoch 进行一次验证
val_cfg = dict(type='ValLoop')

# 测试计划设置：测试时循环的类型
test_cfg = dict(type='TestLoop')

# 学习率调度器设置
param_scheduler = [
    dict(
        type='LinearLR',  # 线性学习率调度器，使用线性函数衰减学习率
        start_factor=0.001,  # 初始学习率因子
        by_epoch=False,  # 根据迭代次数调整学习率
        begin=0,  # 起始迭代次数
        end=500),  # 结束迭代次数
    dict(
        type='MultiStepLR',  # 多步骤学习率调度器，根据指定的里程碑调整学习率
        begin=0,  # 起始迭代次数
        end=12,  # 结束迭代次数
        by_epoch=True,  # 根据 epoch 数调整学习率
        milestones=[8, 11],  # 里程碑，在第 8 和第 11 个 epoch 时调整学习率
        gamma=0.1)  # 学习率调整的倍数因子
]

# 优化器设置
optim_wrapper = dict(
    type='OptimWrapper',  # 优化器包装器
    optimizer=dict(
        type='SGD',  # 随机梯度下降法
        lr=0.02,  # 学习率
        momentum=0.9,  # 动量因子
        weight_decay=0.0001)  # 权重衰减（L2 正则化）因子
)

# 自动调整学习率的默认设置
#   - `enable` 表示是否默认启用自动调整学习率
#   - `base_batch_size` = (8 个 GPU) x (2 个样本数 / GPU)
auto_scale_lr = dict(enable=False, base_batch_size=16)
