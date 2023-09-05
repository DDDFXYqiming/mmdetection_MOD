# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from ..assigners import AssignResult
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@TASK_UTILS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData,
            gt_instances: InstanceData, *args, **kwargs):
        """直接返回正负样本的索引采样结果。

        Args:
            assign_result (:obj:`AssignResult`): bbox分配的结果。
            pred_instances (:obj:`InstanceData`): 模型预测的实例，包括``priors``，
                priors可以是锚点、点或模型预测的bboxes，形状为（n, 4）。
            gt_instances (:obj:`InstanceData`): 实例注释的真实值，通常包括``bboxes``
                和 ``labels`` 属性。

        Returns:
            :obj:`SamplingResult`: 采样结果
        """
        gt_bboxes = gt_instances.bboxes  # 真实边界框
        priors = pred_instances.priors  # 预测实例的priors

        # 正样本索引
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        # 负样本索引
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        # 用于标记priors是否与gt匹配
        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)

        # 创建采样结果对象
        sampling_result = SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False)  # 不使用负样本均值因子

        return sampling_result

