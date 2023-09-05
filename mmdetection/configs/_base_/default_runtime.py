# 默认作用域
default_scope = 'mmdet'

# 默认钩子设置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # 迭代计时器钩子，用于测量每次迭代的时间
    logger=dict(type='LoggerHook', interval=50),  # 日志记录钩子，每 50 个迭代记录一次日志
    param_scheduler=dict(type='ParamSchedulerHook'),  # 参数调度器钩子，用于实现学习率等参数的调度
    checkpoint=dict(type='CheckpointHook', interval=1),  # 模型检查点钩子，每 1 个 epoch 保存一次模型
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 分布式采样器种子钩子，用于设置分布式采样器种子
    visualization=dict(type='DetVisualizationHook')  # 检测结果可视化钩子，用于可视化检测结果
)

# 环境配置
env_cfg = dict(
    cudnn_benchmark=False,  # 是否启用 cudnn 的自动寻优功能
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # 多进程配置
    dist_cfg=dict(backend='nccl'),  # 分布式训练配置，使用 NCCL 后端
)

# 可视化后端配置
vis_backends = [dict(type='LocalVisBackend')]

# 可视化器配置
visualizer = dict(
    type='DetLocalVisualizer',  # 本地可视化器
    vis_backends=vis_backends,  # 可视化后端配置
    name='visualizer'  # 可视化器名称
)

# 日志处理器配置
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# 日志级别设置
log_level = 'INFO'

# 模型加载设置
load_from = None  # 不从预训练模型加载权重
resume = False  # 不从检查点恢复训练
