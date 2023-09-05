# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import List, Optional, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.structures.bbox import (cat_boxes, get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import InstanceList, OptMultiConfig
from ..test_time_augs import merge_aug_results
from ..utils import (filter_scores_and_topk, select_single_mlvl,
                     unpack_gt_instances)


class BaseDenseHead_origin(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads.

    1. The ``init_weights`` method is used to initialize densehead's
    model parameters. After detector initialization, ``init_weights``
    is triggered when ``detector.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of densehead,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict detection results,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict detection results including
    post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    4. The ``loss_and_predict`` method is used to return loss and detection
    results at the same time. It will call densehead's ``forward``,
    ``loss_by_feat`` and ``predict_by_feat`` methods in order.  If one-stage is
    used as RPN, the densehead needs to return both losses and predictions.
    This predictions is used as the proposal of roihead.

    .. code:: text

    loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # `_raw_positive_infos` will be used in `get_positive_infos`, which
        # can get positive information.
        self._raw_positive_infos = dict()

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def get_positive_infos(self) -> InstanceList:
        """Get positive information from sampling results.

        Returns:
            list[:obj:`InstanceData`]: Positive information of each image,
            usually including positive bboxes, positive labels, positive
            priors, etc.
        """
        if len(self._raw_positive_infos) == 0:
            return None

        sampling_results = self._raw_positive_infos.get(
            'sampling_results', None)
        assert sampling_results is not None
        positive_infos = []
        for sampling_result in enumerate(sampling_results):
            pos_info = InstanceData()
            pos_info.bboxes = sampling_result.pos_gt_bboxes
            pos_info.labels = sampling_result.pos_gt_labels
            pos_info.priors = sampling_result.pos_priors
            pos_info.pos_assigned_gt_inds = \
                sampling_result.pos_assigned_gt_inds
            pos_info.pos_inds = sampling_result.pos_inds
            positive_infos.append(pos_info)
        return positive_infos

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """对上游网络的特征进行前向传播和检测头的损失计算。

        Args:
            x (tuple[Tensor]): 来自上游网络的特征，每个特征都是4D张量。
            batch_data_samples (List[:obj:`DetDataSample`]): 数据样本列表，通常包括
                `gt_instance`、`gt_panoptic_seg`和`gt_sem_seg`等信息。

        Returns:
            dict: 损失组成的字典。
        """
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
        batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                            batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    @abstractmethod
    def loss_by_feat(self, **kwargs) -> dict:
        """根据检测头提取的特征计算损失。

        Returns:
            dict: 损失组成的字典。
        """
        pass

    """对检测头进行前向传播，然后根据特征和数据样本计算损失和预测。

    Args:
        x (tuple[Tensor]): FPN提取的特征。
        batch_data_samples (list[:obj:`DetDataSample`]): 每个项包含每个图像的元信息和相应的注释。
        proposal_cfg (ConfigDict, optional): 测试/后处理配置，如果为None，则使用test_cfg。
            默认为None。

    Returns:
        tuple: 返回一个包含以下内容的元组：

            - losses (dict[str, Tensor]): 损失组成的字典。
            - predictions (list[:obj:`InstanceData`]): 经过后处理后的每个图像的检测结果。
    """
    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
        batch_img_metas) = outputs

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                            batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions

    """对上游网络的特征进行检测头的前向传播，并预测检测结果。

    Args:
        x (tuple[Tensor]): 上游网络的多级特征，每个特征都是4D张量。
        batch_data_samples (List[:obj:`DetDataSample`]): 数据样本列表，通常包括
            `gt_instance`、`gt_panoptic_seg`和`gt_sem_seg`等信息。
        rescale (bool, optional): 是否对结果进行重新缩放。默认为False。

    Returns:
        list[obj:`InstanceData`]: 经过后处理后的每个图像的检测结果。
    """
    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    """从头部提取的特征中转换成边界框结果的批量预测函数。

    注意：当score_factors不为None时，通常将cls_scores乘以它以获得在NMS中使用的实际得分，
    例如FCOS中的CenterNess，ATSS中的IoU分支。

    Args:
        cls_scores (list[Tensor]): 所有尺度级别上的分类得分，每个都是一个4D张量，
            形状为(batch_size, num_priors * num_classes, H, W)。
        bbox_preds (list[Tensor]): 所有尺度级别上的边界框能量/偏移，每个都是一个4D张量，
            形状为(batch_size, num_priors * 4, H, W)。
        score_factors (list[Tensor], optional): 所有尺度级别上的得分因子，每个都是一个4D张量，
            形状为(batch_size, num_priors * 1, H, W)。默认为None。
        batch_img_metas (list[dict], optional): 批量图像元信息。默认为None。
        cfg (ConfigDict, optional): 测试/后处理配置，如果为None，则使用test_cfg。默认为None。
        rescale (bool): 如果为True，则返回原始图像空间中的边界框。默认为False。
        with_nms (bool): 如果为True，则在返回边界框之前进行NMS。默认为True。

    Returns:
        list[:obj:`InstanceData`]: 每个图像的目标检测结果经过后处理后的列表。
        每个项通常包含以下键。

            - scores (Tensor): 分类得分，形状为(num_instance, )
            - labels (Tensor): 边界框的标签，形状为(num_instances, )。
            - bboxes (Tensor): 边界框，形状为(num_instances, 4)，
            最后一个维度按照(x1, y1, x2, y2)排列。
    """
    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list


    """将从头部提取的单个图像的特征转换为bbox结果。

    Args:
        cls_score_list (list[Tensor]): 单个图像在所有尺度级别上的分类分数，
            每个项的形状为(num_priors * num_classes, H, W)。
        bbox_pred_list (list[Tensor]): 单个图像在所有尺度级别上的边界框能量/增量，
            每个项的形状为(num_priors * 4, H, W)。
        score_factor_list (list[Tensor]): 单个图像在所有尺度级别上的分数因子，
            每个项的形状为(num_priors * 1, H, W)。
        mlvl_priors (list[Tensor]): 列表中的每个元素是特征金字塔中单个级别的prior。
            在所有基于anchor的方法中，其形状为(num_priors, 4)。
            在所有基于anchor-free的方法中，如果`with_stride=True`，则其形状为(num_priors, 2)，
            否则其形状仍为(num_priors, 4)。
        img_meta (dict): 图像的元信息。
        cfg (mmengine.Config): 测试/后处理配置，如果为None，则使用test_cfg。
        rescale (bool): 如果为True，则返回原始图像空间中的边界框。默认为False。
        with_nms (bool): 如果为True，则在返回边界框之前执行nms。默认为True。

    Returns:
        :obj:`InstanceData`: 后处理后每个图像的检测结果。
        每个项通常包含以下键。

            - scores (Tensor): 分类分数，形状为(num_instance, )
            - labels (Tensor): 边界框的标签，形状为(num_instances, )。
            - bboxes (Tensor): 边界框，形状为(num_instances, 4)，
            最后一个维度的顺序为(x1, y1, x2, y2)。
    """
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        
        if score_factor_list[0] is None:
            # 例如Retina、FreeAnchor等。
            with_score_factors = False
        else:
            # 例如FCOS、PAA、ATSS等。
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                            score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                        0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # 提醒：我们将FG标签设置为[0, num_class-1]
                # 自mmdet v2.0以来
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # 在https://github.com/open-mmlab/mmdetection/pull/6268/之后，
            # 此操作在相同的`nms_pre`下保留更少的边界框。
            # 对于大部分模型，性能没有任何区别。如果您发现性能略有下降，您可以将`nms_pre`设置为比以前更大的值。

            score_thr = cfg.get('score_thr', 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, 'score_factors'):
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]

        return results

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation.

        Args:
            aug_batch_feats (list[tuple[Tensor]]): The outer list
                indicates test-time augmentations and inner tuple
                indicate the multi-level feats from
                FPN, each Tensor should have a shape (B, C, H, W),
            aug_batch_img_metas (list[list[dict]]): Meta information
                of images under the different test-time augs
                (multiscale, flip, etc.). The outer list indicate
                the
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            with_ori_nms (bool): Whether execute the nms in original head.
                Defaults to False. It will be `True` when the head is
                adopted as `rpn_head`.

        Returns:
            list(obj:`InstanceData`): Detection results of the
            input images. Each item usually contains\
            following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances,).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # TODO: remove this for detr and deformdetr
        sig_of_get_results = signature(self.get_results)
        get_results_args = [
            p.name for p in sig_of_get_results.parameters.values()
        ]
        get_results_single_sig = signature(self._get_results_single)
        get_results_single_sig_args = [
            p.name for p in get_results_single_sig.parameters.values()
        ]
        assert ('with_nms' in get_results_args) and \
               ('with_nms' in get_results_single_sig_args), \
               f'{self.__class__.__name__}' \
               'does not support test-time augmentation '

        num_imgs = len(aug_batch_img_metas[0])
        aug_batch_results = []
        for x, img_metas in zip(aug_batch_feats, aug_batch_img_metas):
            outs = self.forward(x)
            batch_instance_results = self.get_results(
                *outs,
                img_metas=img_metas,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=with_ori_nms,
                **kwargs)
            aug_batch_results.append(batch_instance_results)

        # after merging, bboxes will be rescaled to the original image
        batch_results = merge_aug_results(aug_batch_results,
                                          aug_batch_img_metas)

        final_results = []
        for img_id in range(num_imgs):
            results = batch_results[img_id]
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores,
                                                results.labels,
                                                self.test_cfg.nms)
            results = results[keep_idxs]
            # some nms operation may reweight the score such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:self.test_cfg.max_per_img]
            if rescale:
                # all results have been mapped to the original scale
                # in `merge_aug_results`, so just pass
                pass
            else:
                # map to the first aug image scale
                scale_factor = results.bboxes.new_tensor(
                    aug_batch_img_metas[0][img_id]['scale_factor'])
                results.bboxes = \
                    results.bboxes * scale_factor

            final_results.append(results)

        return final_results
