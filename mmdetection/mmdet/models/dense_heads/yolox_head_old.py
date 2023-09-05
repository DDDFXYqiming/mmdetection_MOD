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
class YOLOXHead(BaseDenseHead):
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

        loss_kp: ConfigType = dict(
            type="SmoothL1Loss", beta=1.0, reduction="sum", loss_weight=1.0
        ),  # 定义关键点损失

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

        self.loss_cls: nn.Module = MODELS.build(loss_cls)
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)
        self.loss_obj: nn.Module = MODELS.build(loss_obj)
        self.loss_kp: nn.Module = MODELS.build(loss_kp)  # 构建关键点损失函数

        self.use_l1 = False
        self.loss_l1: nn.Module = MODELS.build(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg["assigner"])
            self.sampler = PseudoSampler()

        self._init_layers()

    def _init_layers(self) -> None:
        """为所有级别的特征图初始化头部。"""
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        self.multi_level_conv_kp1 = nn.ModuleList()  # 新增关键点预测器
        self.multi_level_conv_kp2 = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())  # 构建分类卷积层
            self.multi_level_reg_convs.append(self._build_stacked_convs())  # 构建定位卷积层
            (
                conv_cls,
                conv_reg,
                conv_obj,
                conv_kp1,
                conv_kp2,
            ) = self._build_predictor()  # 构建预测器
            self.multi_level_conv_cls.append(conv_cls)  # 分类预测器
            self.multi_level_conv_reg.append(conv_reg)  # 定位预测器
            self.multi_level_conv_obj.append(conv_obj)  # 目标性预测器
            self.multi_level_conv_kp1.append(conv_kp1)  # 第一个关键点预测器
            self.multi_level_conv_kp2.append(conv_kp2)  # 第二个关键点预测器

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

    def _build_predictor(
        self,
    ) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
        """初始化单个级别头部的预测器层。"""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)  # 分类预测
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)  # 定位预测
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)  # 目标性预测
        conv_kp1 = nn.Conv2d(self.feat_channels, 2, 1)  # 第一个关键点预测
        conv_kp2 = nn.Conv2d(self.feat_channels, 2, 1)  # 第二个关键点预测
        return conv_cls, conv_reg, conv_obj, conv_kp1, conv_kp2

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
        conv_kp1: nn.Module,
        conv_kp2: nn.Module,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        kp1_pred = conv_kp1(reg_feat)  # 第一个关键点预测
        kp2_pred = conv_kp2(reg_feat)  # 第二个关键点预测

        return cls_score, bbox_pred, objectness, kp1_pred, kp2_pred

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
            self.multi_level_conv_kp1,  # 新增参数
            self.multi_level_conv_kp2,  # 新增参数
        )

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
    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        objectnesses: Optional[List[Tensor]],
        kp1_preds: List[Tensor],
        kp2_preds: List[Tensor],
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses) == len(kp1_preds) == len(kp2_preds)
        cfg = self.test_cfg if cfg is None else cfg

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True,
        )

        # flatten cls_scores, bbox_preds, objectness, kp1_preds, and kp2_preds
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
        flatten_kp1_preds = [
            kp1_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for kp1_pred in kp1_preds
        ]
        flatten_kp2_preds = [
            kp2_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for kp2_pred in kp2_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_kp1_preds = torch.cat(flatten_kp1_preds, dim=1)
        flatten_kp2_preds = torch.cat(flatten_kp2_preds, dim=1)

        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        flatten_kp1 = self.decode_keypoints(flatten_priors,flatten_kp1_preds)
        flatten_kp2 = self.decode_keypoints(flatten_priors,flatten_kp2_preds)


        result_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            max_scores, labels = torch.max(flatten_cls_scores[img_id], 1)
            valid_mask = flatten_objectness[img_id] * max_scores >= cfg.score_thr
            results = InstanceData(
                bboxes=flatten_bboxes[img_id][valid_mask],
                scores=max_scores[valid_mask] * flatten_objectness[img_id][valid_mask],
                labels=labels[valid_mask],
                keypoints1=flatten_kp1[img_id][valid_mask],  # 添加关键点信息
                keypoints2=flatten_kp2[img_id][valid_mask],
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
    
    def decode_keypoints(self, priors: Tensor, kp_preds: Tensor) -> Tensor:
        decoded_keypoints = (kp_preds * priors[:, 2:]) + priors[:, :2]

        return decoded_keypoints


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
        if rescale:
            assert img_meta.get("scale_factor") is not None
            a = results.bboxes.new_tensor(
                img_meta["scale_factor"]
            ).repeat((1, 2))

            results.bboxes /= a
            results.keypoints1 /= results.keypoints1.new_tensor(
                img_meta["scale_factor"]
            ).repeat((1, 1))
            results.keypoints2 /= results.keypoints2.new_tensor(
                img_meta["scale_factor"]
            ).repeat((1, 1))

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
        kp1_preds: Sequence[Tensor],  # 第一个关键点预测结果
        kp2_preds: Sequence[Tensor],  # 第二个关键点预测结果
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
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
        flatten_kp1_preds = [
            kp1_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for kp1_pred in kp1_preds
        ]
        flatten_kp2_preds = [
            kp2_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for kp2_pred in kp2_preds
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_kp1_preds = torch.cat(flatten_kp1_preds, dim=1)
        flatten_kp2_preds = torch.cat(flatten_kp2_preds, dim=1)
        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        flatten_kp1 = self.decode_keypoints(flatten_priors,flatten_kp1_preds)
        flatten_kp2 = self.decode_keypoints(flatten_priors,flatten_kp2_preds)

        (
            pos_masks,
            cls_targets,
            obj_targets,
            bbox_targets,
            l1_targets,
            kp1_targets,
            kp2_targets,
            num_fg_imgs,
        ) = multi_apply(
            self._get_targets_single,
            flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
            flatten_cls_preds.detach(),
            flatten_bboxes.detach(),
            flatten_objectness.detach(), 
            flatten_kp1.detach(),
            flatten_kp2.detach(),
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
        kp1_targets = torch.cat(kp1_targets, 0)
        kp2_targets = torch.cat(kp2_targets, 0)

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

            # 创建有效关键点掩码
            valid_mask = (kp1_targets > 0).all(dim = 1) & (kp2_targets > 0).all(dim = 1)

            # 新增关键点的损失计算
            # print(flatten_kp1_preds.view(-1, 2)[pos_masks])
            # print(kp1_targets)
            loss_kp1 = (
                self.loss_kp(flatten_kp1_preds.view(-1, 2)[pos_masks][valid_mask], kp1_targets[valid_mask])
                /  (valid_mask.sum().item() + 1e-8)
            )
            loss_kp2 = (
                self.loss_kp(flatten_kp2_preds.view(-1, 2)[pos_masks][valid_mask], kp2_targets[valid_mask])
                /  (valid_mask.sum().item() + 1e-8)
            )
            # if kp1_targets[0] == [0.,0.,0.] or kp2_targets[0] == [0.,0.,0.]:
            #     loss_kp1 = flatten_kp1_preds.sum() * 0
            #     loss_kp2 = flatten_kp2_preds.sum() * 0

        else:
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0
            loss_kp1 = flatten_kp1_preds.sum() * 0
            loss_kp2 = flatten_kp2_preds.sum() * 0

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_obj=loss_obj,
            loss_kp1=loss_kp1,
            loss_kp2=loss_kp2,
        )

        if self.use_l1:
            if num_pos > 0:
                loss_l1 = (
                    self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks], l1_targets)
                    / num_total_samples
                )
            else:
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
        kp1_preds: Tensor,  # 新增参数，关键点1的预测
        kp2_preds: Tensor,  # 新增参数，关键点2的预测
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: Optional[InstanceData] = None,
    ) -> tuple:
        num_priors = priors.size(0)
        num_gts = len(gt_instances)
        # 没有目标
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            kp1_targets = cls_preds.new_zeros((0, 2))  # 添加关键点目标设置为零张量
            kp2_targets = cls_preds.new_zeros((0, 2))  # 添加关键点目标设置为零张量
            return (
                foreground_mask,
                cls_target,
                obj_target,
                bbox_target,
                l1_target,
                kp1_targets,
                kp2_targets,
                0,
            )

        # YOLOX使用带有0.5偏移的中心prior分配目标，
        # 但在回归bbox时使用没有偏移的中心prior。
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1
        )
        
        # 计算预测目标的IOU感知分类得分
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

        # 计算目标检测任务中的目标、边界框等其他训练目标
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1

        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])

        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1

        num_gts = len(gt_instances)
        num_kps = 2  # 假设每个目标有2个关键点（x, y）坐标

        # 获取真实关键点坐标
        # print(gt_instances)
        # if hasattr(gt_instances, 'keypoints'):
        gt_kps = gt_instances.keypoints  # 假设InstanceData中有保存真实关键点的坐标

        # 计算关键点目标
        kp1_targets = cls_preds.new_zeros((num_pos_per_img, 2))  # 修改关键点目标的维度
        kp2_targets = cls_preds.new_zeros((num_pos_per_img, 2))  # 修改关键点目标的维度

        for i in range(num_pos_per_img):
            gt_idx = sampling_result.pos_assigned_gt_inds[i]
            if gt_idx >= 0:
                keypoints = gt_kps[gt_idx, :, :2]  # 提取前两列坐标信息
                kp1_targets[i] = keypoints[0]  # 赋值给kp1_targets
                kp2_targets[i] = keypoints[1]
        # else:
        #     kp1_targets = cls_preds.new_zeros((num_pos_per_img, 2))  # 添加关键点目标设置为零张量
        #     kp2_targets = cls_preds.new_zeros((num_pos_per_img, 2))  # 添加关键点目标设置为零张量


        return (
            foreground_mask,
            cls_target,
            obj_target,
            bbox_target,
            l1_target,
            kp1_targets,
            kp2_targets,
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
