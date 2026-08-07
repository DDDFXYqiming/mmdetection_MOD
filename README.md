# mmdetection_MOD 数据管线（mmdet-scripts）

多类别目标检测 + 关键点估计项目（YOLOX / mmdetection）。仓库主体为官方 mmdetection 框架，本仓库 `refactor` 分支将**顶层原创脚本**重构为统一数据管线包，框架本体未做任何改动。

## 原脚本 → 新命令映射

| 原脚本 | 新命令 |
| --- | --- |
| `to_coco.py` | `mmdet-scripts convert --variant keypoints` + `split` |
| `convert_to_coco.py` | `mmdet-scripts convert --variant standard` |
| `mini_coco_created.py` | `mmdet-scripts convert --variant mini` |
| `to_coco_point_to_box_test.py` | `mmdet-scripts convert --variant point-box` |
| `split_image.py` / `to_coco*.py` 内 split | `mmdet-scripts split` |
| `divide_into_small_camera.py` | `mmdet-scripts batch` |
| `choose_by_labels_and_distance.py` | `mmdet-scripts select` |
| `match_image.py` / `delete_json.py` | `mmdet-scripts cleanup` |
| `visible.py` / `visible_old.py` | `mmdet-scripts visualize` |
| `pkl_cv.py` / `combined_cv.py` | `mmdet-scripts viz-pkl` |
| `mp4.py` | `mmdet-scripts to-video` |
| `read_pkl.py` | `mmdet-scripts read-pkl` |
| `pth2onnx.py` | `mmdet-scripts export-onnx`（需要 torch + mmdetection 环境） |

## 安装

```bash
python -m pip install -e .
```

依赖：Python 3.9+，opencv-python-headless（转换/分割/筛选类命令只需标准库）。

## 命令示例

```bash
# 按时间戳分组分割数据集（整组进入同一集合，避免连续帧拆散）
mmdet-scripts split J2camera_selected_50 mmdetection/data --train-ratio 0.9 --val-ratio 0.05 --max-time-diff 120

# 转换为 COCO 标注（4 类 + 关键点）
mmdet-scripts convert mmdetection/data/coco/train2017 mmdetection/data/coco/annotations --output instances_train2017.json --variant keypoints

# 按标签 + 距离筛选（默认修正后的距离公式）
mmdet-scripts select J2camera J2camera_selected_50 --max-distance 50 --interval 4

# 清理无图 json / coco 中间 json
mmdet-scripts cleanup J2camera --orphan-json
mmdet-scripts cleanup mmdetection/data/coco --coco-json

# 可视化
mmdet-scripts visualize instances_val2017.json mmdetection/data/coco/val2017 --output annotated_images
mmdet-scripts viz-pkl results.pkl --output pkl_cv
mmdet-scripts viz-pkl results.pkl --compare --annotation-file instances_val2017.json --image-folder mmdetection/data/coco/val2017

# 图片序列 → 视频
mmdet-scripts to-video mmdetection/data/coco/val2017 --output demo_car.mp4 --fps 10

# 查看 pkl
mmdet-scripts read-pkl results.pkl --limit 3

# pth → onnx（需 torch）
mmdet-scripts export-onnx epoch_300.pth --output yolox.onnx
```

## 重要修正

1. **距离公式**（`select`）：原版行人/骑行距离公式分子分母同时含像素高度，化简后与图像高度无关（笔误）。重构默认使用小孔成像正确公式 `distance = 物体实际尺寸 × 焦距 ÷ 像素尺寸`；需要复现原行为可用 `--formula legacy`。
2. **面积计算**（`convert`）：原版 `area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])` 用宽高差计算，实际应为 `宽 × 高`，已修正。
3. **分割分组**（`split`）：原 `split_image.py` 的测试集分支存在缩进错误（组外引用未定义变量），已统一为正确的整组分配逻辑。
4. 所有路径/比例/阈值改为命令行参数，不再硬编码。

## 目录结构

```text
src/mmdet_scripts/
  cli.py          # 统一命令行入口
  config.py       # 类别映射与默认参数
  utils.py        # 文件/标注通用工具
  coco/           # split（分割/分批）、convert（COCO 转换）、cleanup（清理）
  filter/         # select（按标签+距离筛选）
  visualize/      # annotations（标注可视化）、predictions（pkl 可视化/对比）
  tools/          # media（视频）、pkl_inspect（pkl 查看）
  export/         # onnx（pth → onnx）
```
