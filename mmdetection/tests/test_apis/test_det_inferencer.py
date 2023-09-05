# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase, mock
from unittest.mock import Mock, patch

import mmcv
import mmengine
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmengine.utils import is_list_of
from parameterized import parameterized

from mmdet.apis import DetInferencer
from mmdet.evaluation.functional import get_classes
from mmdet.structures import DetDataSample

# 创建一个测试类 TestDetInferencer，继承自 TestCase 类
class TestDetInferencer(TestCase):

    # 使用 mock.patch 装饰器，模拟 _load_checkpoint 函数的返回值为 None，用于测试 init 函数
    @mock.patch('mmengine.infer.infer._load_checkpoint', return_value=None)
    def test_init(self, mock):
        # 通过 metafile 初始化 DetInferencer
        DetInferencer('rtmdet-t')
        # 通过配置文件初始化 DetInferencer
        DetInferencer('configs/yolox/yolox_tiny_8xb8-300e_coco.py')

    # 定义一个自定义的函数，用于断言两个预测结果是否相等
    def assert_predictions_equal(self, preds1, preds2):
        for pred1, pred2 in zip(preds1, preds2):
            if 'bboxes' in pred1:
                self.assertTrue(
                    np.allclose(pred1['bboxes'], pred2['bboxes'], 0.1))
            if 'scores' in pred1:
                self.assertTrue(
                    np.allclose(pred1['scores'], pred2['scores'], 0.1))
            if 'labels' in pred1:
                self.assertTrue(np.allclose(pred1['labels'], pred2['labels']))
            if 'panoptic_seg_path' in pred1:
                self.assertTrue(
                    pred1['panoptic_seg_path'] == pred2['panoptic_seg_path'])

    # 使用 parameterized.expand 装饰器，定义一个测试函数，用于测试 call 函数
    @parameterized.expand([
        'rtmdet-t', 'mask-rcnn_r50_fpn_1x_coco', 'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_call(self, model):
        # 测试单张图片
        img_path = 'tests/data/color.jpg'

        mock_load = Mock(return_value=None)
        with patch('mmengine.infer.infer._load_checkpoint', mock_load):
            inferencer = DetInferencer(model)

        # 在不加载预训练权重的情况下，默认类别为 COCO 80，需要进行替换
        if model == 'panoptic_fpn_r50_fpn_1x_coco':
            inferencer.visualizer.dataset_meta = {
                'classes': get_classes('coco_panoptic'),
                'palette': 'random'
            }

        res_path = inferencer(img_path, return_vis=True)
        # ndarray
        img = mmcv.imread(img_path)
        res_ndarray = inferencer(img, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # 测试多张图片
        img_paths = ['tests/data/color.jpg', 'tests/data/gray.jpg']
        res_path = inferencer(img_paths, return_vis=True)
        # list of ndarray
        imgs = [mmcv.imread(p) for p in img_paths]
        res_ndarray = inferencer(imgs, return_vis=True)
        self.assert_predictions_equal(res_path['predictions'],
                                      res_ndarray['predictions'])
        self.assertIn('visualization', res_path)
        self.assertIn('visualization', res_ndarray)

        # 图像文件夹，测试不同的 batch 大小
        img_dir = 'tests/data/VOCdevkit/VOC2007/JPEGImages/'
        res_bs1 = inferencer(img_dir, batch_size=1, return_vis=True)
        res_bs3 = inferencer(img_dir, batch_size=3, return_vis=True)
        self.assert_predictions_equal(res_bs1['predictions'],
                                      res_bs3['predictions'])

        # 在绘制 mask 时有抖动操作，所以无法进行断言
        if model == 'rtmdet-t':
            for res_bs1_vis, res_bs3_vis in zip(res_bs1['visualization'],
                                                res_bs3['visualization']):
                self.assertTrue(np.allclose(res_bs1_vis, res_bs3_vis))

    # 使用 parameterized.expand 装饰器，定义一个测试函数，用于测试 visualize 函数
    @parameterized.expand([
        'rtmdet-t', 'mask-rcnn_r50_fpn_1x_coco', 'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_visualize(self, model):
        img_paths = ['tests/data/color.jpg', 'tests/data/gray.jpg']

        mock_load = Mock(return_value=None)
        with patch('mmengine.infer.infer._load_checkpoint', mock_load):
            inferencer = DetInferencer(model)

        # 在不加载预训练权重的情况下，默认类别为 COCO 80，需要进行替换
        if model == 'panoptic_fpn_r50_fpn_1x_coco':
            inferencer.visualizer.dataset_meta = {
                'classes': get_classes('coco_panoptic'),
                'palette': 'random'
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            inferencer(img_paths, out_dir=tmp_dir)
            for img_dir in ['color.jpg', 'gray.jpg']:
                self.assertTrue(osp.exists(osp.join(tmp_dir, 'vis', img_dir)))

    # 使用 parameterized.expand 装饰器，定义一个测试函数，用于测试 postprocess 函数
    @parameterized.expand([
        'rtmdet-t', 'mask-rcnn_r50_fpn_1x_coco', 'panoptic_fpn_r50_fpn_1x_coco'
    ])
    def test_postprocess(self, model):
        # return_datasample
        img_path = 'tests/data/color.jpg'

        mock_load = Mock(return_value=None)
        with patch('mmengine.infer.infer._load_checkpoint', mock_load):
            inferencer = DetInferencer(model)

        # 在不加载预训练权重的情况下，默认类别为 COCO 80，需要进行替换
        if model == 'panoptic_fpn_r50_fpn_1x_coco':
            inferencer.visualizer.dataset_meta = {
                'classes': get_classes('coco_panoptic'),
                'palette': 'random'
            }

        res = inferencer(img_path, return_datasample=True)
        self.assertTrue(is_list_of(res['predictions'], DetDataSample))

        with tempfile.TemporaryDirectory() as tmp_dir:
            res = inferencer(img_path, out_dir=tmp_dir, no_save_pred=False)
            dumped_res = mmengine.load(
                osp.join(tmp_dir, 'preds', 'color.json'))
            self.assertEqual(res['predictions'][0], dumped_res)

    # 使用 mock.patch 装饰器，模拟 _load_checkpoint 函数的返回值为 None，用于测试 pred2dict 函数
    @mock.patch('mmengine.infer.infer._load_checkpoint', return_value=None)
    def test_pred2dict(self, mock):
        data_sample = DetDataSample()
        data_sample.pred_instances = InstanceData()

        data_sample.pred_instances.bboxes = np.array([[0, 0, 1, 1]])
        data_sample.pred_instances.labels = np.array([0])
        data_sample.pred_instances.scores = torch.FloatTensor([0.9])
        res = DetInferencer('rtmdet-t').pred2dict(data_sample)
        self.assertListAlmostEqual(res['bboxes'], [[0, 0, 1, 1]])
        self.assertListAlmostEqual(res['labels'], [0])
        self.assertListAlmostEqual(res['scores'], [0.9])

    # 自定义函数，用于断言两个列表中的元素是否近似相等
    def assertListAlmostEqual(self, list1, list2, places=7):
        for i in range(len(list1)):
            if isinstance(list1[i], list):
                self.assertListAlmostEqual(list1[i], list2[i], places=places)
            else:
                self.assertAlmostEqual(list1[i], list2[i], places=places)

