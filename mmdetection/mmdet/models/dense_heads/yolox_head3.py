# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.ops.nms import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import (
    ConfigType,
    OptConfigType,
    OptInstanceList,
    OptMultiConfig,
    reduce_mean,
)
from ..task_modules.prior_generators import MlvlPointGenerator
from ..task_modules.samplers import PseudoSampler
from ..utils import multi_apply
from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class YOLOXHead3(BaseDenseHead):
    """`YOLOX <https://arxiv.org/abs/2107.08430>`_算法中使用的YOLOXHead头部。

    Args:
        num_classes (int): 不包括背景类别的类别数量。
        in_channels (int): 输入特征图的通道数。
        feat_channels (int): 堆叠卷积中的隐藏通道数。默认为256。
        stacked_convs (int): 头部中堆叠卷积的数量。默认为(8, 16, 32)。
        strides (Sequence[int]): 每个特征图的下采样因子。默认为None。
        use_depthwise (bool): 是否在块中使用深度可分离卷积。默认为False。
        dcn_on_last_conv (bool): 如果为True，在towers的最后一层使用DCN。默认为False。
        conv_bias (bool or str): 如果指定为`auto`，则由norm_cfg决定。如果`norm_cfg`为None，则卷积的偏置将设置为True，否则为False。默认为"auto"。
        conv_cfg (:obj:`ConfigDict` or dict, optional): 卷积层的配置字典。默认为None。
        norm_cfg (:obj:`ConfigDict` or dict): 归一化层的配置字典。默认为dict(type='BN', momentum=0.03, eps=0.001)。
        act_cfg (:obj:`ConfigDict` or dict): 激活层的配置字典。默认为None。
        loss_cls (:obj:`ConfigDict` or dict): 分类损失的配置。
        loss_bbox (:obj:`ConfigDict` or dict): 定位损失的配置。
        loss_obj (:obj:`ConfigDict` or dict): 目标性损失的配置。
        loss_l1 (:obj:`ConfigDict` or dict): L1损失的配置。
        train_cfg (:obj:`ConfigDict` or dict, optional): anchor head的训练配置。默认为None。
        test_cfg (:obj:`ConfigDict` or dict, optional): anchor head的测试配置。默认为None。
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): 初始化配置字典。默认为None。
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        strides: Sequence[int] = (8, 16, 32),
        use_depthwise: bool = False,
        dcn_on_last_conv: bool = False,
        conv_bias: Union[bool, str] = "auto",
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="Swish"),
        loss_cls: ConfigType = dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0
        ),
        loss_bbox: ConfigType = dict(
            type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0
        ),
        loss_obj: ConfigType = dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0
        ),
        loss_l1: ConfigType = dict(type="L1Loss", reduction="sum", loss_weight=1.0),
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == "auto" or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls: nn.Module = MODELS.build(loss_cls)  # 构建分类损失函数
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)  # 构建定位损失函数
        self.loss_obj: nn.Module = MODELS.build(loss_obj)  # 构建目标性损失函数

        self.use_l1 = False  # 这个标志位会在钩子中被修改
        self.loss_l1: nn.Module = MODELS.build(loss_l1)  # 构建L1损失函数

        self.prior_generator = MlvlPointGenerator(strides, offset=0)  # 多级点生成器

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg["assigner"])  # 构建目标分配器
            # YOLOX不支持采样
            self.sampler = PseudoSampler()

        self._init_layers()  # 初始化网络层

    def _init_layers(self) -> None:
        """为所有级别的特征图初始化头部。"""
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())  # 构建分类卷积层
            self.multi_level_reg_convs.append(self._build_stacked_convs())  # 构建定位卷积层
            conv_cls, conv_reg, conv_obj = self._build_predictor()  # 构建预测器
            self.multi_level_conv_cls.append(conv_cls)  # 分类预测器
            self.multi_level_conv_reg.append(conv_reg)  # 定位预测器
            self.multi_level_conv_obj.append(conv_obj)  # 目标性预测器

    def _build_stacked_convs(self) -> nn.Sequential:
        """初始化单个级别头部的卷积层。"""
        conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type="DCNv2")
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias,
                )
            )
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """初始化单个级别头部的预测器层。"""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)  # 分类预测
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)  # 定位预测
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)  # 目标性预测
        return conv_cls, conv_reg, conv_obj

    def init_weights(self) -> None:
        """初始化头部的权重。"""
        super(YOLOXHead, self).init_weights()
        # 在模型初始化中使用prior提高稳定性
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(
            self.multi_level_conv_cls, self.multi_level_conv_obj
        ):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward_single(
        self,
        x: Tensor,
        cls_convs: nn.Module,
        reg_convs: nn.Module,
        conv_cls: nn.Module,
        conv_reg: nn.Module,
        conv_obj: nn.Module,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """单个尺度级别的特征前向传播。

        Args:
            x (Tensor): 输入特征图。
            cls_convs (nn.Module): 分类头部的卷积层。
            reg_convs (nn.Module): 定位头部的卷积层。
            conv_cls (nn.Module): 分类预测器。
            conv_reg (nn.Module): 定位预测器。
            conv_obj (nn.Module): 目标性预测器。

        Returns:
            tuple: 分类预测，定位预测，目标性预测。
        """

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)  # 分类预测
        bbox_pred = conv_reg(reg_feat)  # 定位预测
        objectness = conv_obj(reg_feat)  # 目标性预测

        return cls_score, bbox_pred, objectness

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """从上游网络前向传播特征。

        Args:
            x (Tuple[Tensor]): 来自上游网络的特征，每个都是一个4D张量。

        Returns:
            Tuple[List]: 包含多级分类分数、bbox预测和目标性预测的元组。
        """

        return multi_apply(
            self.forward_single,
            x,
            self.multi_level_cls_convs,
            self.multi_level_reg_convs,
            self.multi_level_conv_cls,
            self.multi_level_conv_reg,
            self.multi_level_conv_obj,
        )

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        objectnesses: Optional[List[Tensor]],
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> List[InstanceData]:
        """将头部提取的一批输出特征转换为bbox结果。

        Args:
            cls_scores (list[Tensor]): 所有尺度级别的分类分数，每个都是一个4D张量，形状为(batch_size, num_priors * num_classes, H, W)。
            bbox_preds (list[Tensor]): 所有尺度级别的盒子能量/偏差，每个都是一个4D张量，形状为(batch_size, num_priors * 4, H, W)。
            objectnesses (list[Tensor], Optional): 所有尺度级别的目标性得分，每个都是一个4D张量，形状为(batch_size, 1, H, W)。
            batch_img_metas (list[dict], Optional): 批图像的元信息。默认为None。
            cfg (ConfigDict, optional): 测试/后处理配置，如果为None，将使用test_cfg。默认为None。
            rescale (bool): 如果为True，则在原始图像空间中返回框。默认为False。
            with_nms (bool): 如果为True，在返回框之前进行NMS。默认为True。

        Returns:
            list[:obj:`InstanceData`]: 经过后处理的每个图像的目标检测结果。每个项目通常包含以下键。

            - scores (Tensor): 分类分数，形状为(num_instance, )
            - labels (Tensor): 边界框的标签，形状为(num_instances, )。
            - bboxes (Tensor): 形状为(num_instances, 4)的边界框，最后一个维度4为(x1，y1，x2，y2)。
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True,
        )

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            max_scores, labels = torch.max(flatten_cls_scores[img_id], 1)
            valid_mask = flatten_objectness[img_id] * max_scores >= cfg.score_thr
            results = InstanceData(
                bboxes=flatten_bboxes[img_id][valid_mask],
                scores=max_scores[valid_mask] * flatten_objectness[img_id][valid_mask],
                labels=labels[valid_mask],
            )

            result_list.append(
                self._bbox_post_process(
                    results=results,
                    cfg=cfg,
                    rescale=rescale,
                    with_nms=with_nms,
                    img_meta=img_meta,
                )
            )

        return result_list

    def _bbox_decode(self, priors: Tensor, bbox_preds: Tensor) -> Tensor:
        """将回归结果（delta_x、delta_x、w、h）解码为边界框（tl_x、tl_y、br_x、br_y）。

        Args:
            priors (Tensor): 图像的中心priors，形状为(num_instances, 2)。
            bbox_preds (Tensor): 所有实例的盒子能量/偏差，形状为(batch_size, num_instances, 4)。

        Returns:
            Tensor: 解码后的边界框，格式为(tl_x、tl_y、br_x、br_y)。形状为(batch_size, num_instances, 4)。
        """
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def _bbox_post_process(
        self,
        results: InstanceData,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: Optional[dict] = None,
    ) -> InstanceData:
        """bbox后处理方法。

        对框进行rescale到原始图像尺度，并进行nms操作。通常with_nms为False用于增强测试。

        Args:
            results (:obj:`InstaceData`): 检测实例结果，每个项目形状为(num_bboxes, )。
            cfg (mmengine.Config): 测试/后处理配置，如果为None，将使用test_cfg。
            rescale (bool): 如果为True，则在原始图像空间中返回框。默认为False。
            with_nms (bool): 如果为True，在返回框之前进行NMS。默认为True。
            img_meta (dict, optional): 图像元信息。默认为None。

        Returns:
            :obj:`InstanceData`: 经过后处理的每个图像的检测结果。
            每个项目通常包含以下键。

            - scores (Tensor): 分类分数，形状为(num_instance, )
            - labels (Tensor): 边界框的标签，形状为(num_instances, )。
            - bboxes (Tensor): 形状为(num_instances, 4)的边界框，最后一个维度4为(x1，y1，x2，y2)。
        """
        if rescale:
            assert img_meta.get("scale_factor") is not None
            results.bboxes /= results.bboxes.new_tensor(
                img_meta["scale_factor"]
            ).repeat((1, 2))

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(
                results.bboxes, results.scores, results.labels, cfg.nms
            )
            results = results[keep_idxs]
            # 一些nms会重新加权得分，例如softnms
            results.scores = det_bboxes[:, -1]
        return results

    def loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        objectnesses: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """根据检测头提取的特征计算损失。

        Args:
            cls_scores (Sequence[Tensor]): 每个尺度级别的分类得分，每个都是一个4D张量，通道数为num_priors * num_classes。
            bbox_preds (Sequence[Tensor]): 每个尺度级别的盒子能量/偏差，每个都是一个4D张量，通道数为num_priors * 4。
            objectnesses (Sequence[Tensor]): 所有尺度级别的目标性得分，每个都是一个4D张量，形状为(batch_size, 1, H, W)。
            batch_gt_instances (list[:obj:`InstanceData`]): 一批gt_instance。通常包括'bboxes'和'labels'属性。
            batch_img_metas (list[dict]): 每个图像的元信息，例如图像尺寸、缩放因子等。
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional): 被忽略的gt_instances的批处理。它包含要在训练和测试中忽略的'bboxes'属性数据。默认为None。

        Returns:
            dict[str, Tensor]: 包含损失的字典。
        """
        num_imgs = len(batch_img_metas)
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True,
        )

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (
            pos_masks,
            cls_targets,
            obj_targets,
            bbox_targets,
            l1_targets,
            num_fg_imgs,
        ) = multi_apply(
            self._get_targets_single,
            flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
            flatten_cls_preds.detach(),
            flatten_bboxes.detach(),
            flatten_objectness.detach(),
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )

        # 实验结果表明，'reduce_mean'可以提高COCO数据集上的性能。
        num_pos = torch.tensor(
            sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_preds.device
        )
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_obj = (
            self.loss_obj(flatten_objectness.view(-1, 1), obj_targets)
            / num_total_samples
        )
        if num_pos > 0:
            loss_cls = (
                self.loss_cls(
                    flatten_cls_preds.view(-1, self.num_classes)[pos_masks], cls_targets
                )
                / num_total_samples
            )
            loss_bbox = (
                self.loss_bbox(flatten_bboxes.view(-1, 4)[pos_masks], bbox_targets)
                / num_total_samples
            )
        else:
            # 避免在图像中没有ground-truth时，cls和reg分支不参与梯度传播。
            # 更多细节，请参见
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            if num_pos > 0:
                loss_l1 = (
                    self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks], l1_targets)
                    / num_total_samples
                )
            else:
                # 避免在图像中没有ground-truth时，cls和reg分支不参与梯度传播。
                # 更多细节，请参见
                # https://github.com/open-mmlab/mmdetection/issues/7298
                loss_l1 = flatten_bbox_preds.sum() * 0
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    @torch.no_grad()
    def _get_targets_single(
        self,
        priors: Tensor,
        cls_preds: Tensor,
        decoded_bboxes: Tensor,
        objectness: Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: Optional[InstanceData] = None,
    ) -> tuple:
        """为单个图像中的priors计算分类、回归和目标性目标。

        Args:
            priors (Tensor): 一个图像的所有priors，形状为[num_priors, 4]的2D张量，格式为[cx，xy，stride_w，stride_y]。
            cls_preds (Tensor): 一个图像的分类预测，形状为[num_priors，num_classes]的2D张量。
            decoded_bboxes (Tensor): 一个图像的解码bbox预测，形状为[num_priors，4]的2D张量，格式为[tl_x，tl_y，br_x，br_y]。
            objectness (Tensor): 一个图像的目标性预测，形状为[num_priors]的1D张量。
            gt_instances (:obj:`InstanceData`): 实例注释的ground truth。它应该包括“bboxes”和“labels”属性。
            img_meta (dict): 当前图像的元信息。
            gt_instances_ignore (:obj:`InstanceData`, optional): 在训练时要忽略的实例。它包含在训练和测试中忽略的“bboxes”属性数据。默认为None。

        Returns:
            tuple:
                foreground_mask (list[Tensor]): 前景目标的二进制掩码。
                cls_target (list[Tensor]): 图像的分类目标。
                obj_target (list[Tensor]): 图像的目标性目标。
                bbox_target (list[Tensor]): 图像的bbox目标。
                l1_target (int): 图像的bbox L1目标。
                num_pos_per_img (int): 图像中的正样本数。
        """

        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # 没有目标
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target, l1_target, 0)

        # YOLOX使用带有0.5偏移的中心prior分配目标，
        # 但在回归bbox时使用没有偏移的中心prior。
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1
        )

        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(
            bboxes=decoded_bboxes, scores=scores.sqrt_(), priors=offset_priors
        )
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore,
        )

        sampling_result = self.sampler.sample(
            assign_result, pred_instances, gt_instances
        )
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU感知分类得分
        cls_target = F.one_hot(
            sampling_result.pos_gt_labels, self.num_classes
        ) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (
            foreground_mask,
            cls_target,
            obj_target,
            bbox_target,
            l1_target,
            num_pos_per_img,
        )

    def _get_l1_target(
        self, l1_target: Tensor, gt_bboxes: Tensor, priors: Tensor, eps: float = 1e-8
    ) -> Tensor:
        """将gt bbox转换为中心偏移和log宽高。"""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
