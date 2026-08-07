"""原始标注 json → COCO 格式（多种变体）。"""

import os
from typing import Callable

from ..config import KEYPOINT_CATEGORIES, OBS_TYPE_TO_CATEGORY
from ..utils import (
    iter_json_files,
    load_json,
    rect_to_bbox,
    rect_to_segmentation,
    save_json,
)


def _category_name(obs_type: str) -> str:
    return obs_type.lower().replace("obstaclerawmodel_", "")


def _build_categories(obs_map: dict[str, int], with_keypoints: bool = False) -> list[dict]:
    categories = []
    for obs_type, category_id in obs_map.items():
        keypoints = []
        skeleton = []
        if with_keypoints and _category_name(obs_type) in ("fullcar", "cyclist"):
            keypoints = ["front", "rear"]
        categories.append(
            {
                "id": category_id,
                "name": _category_name(obs_type),
                "supercategory": "Vehicle",
                **({"keypoints": keypoints, "skeleton": skeleton} if with_keypoints else {}),
            }
        )
    return categories


def _parse_image(data: list) -> dict | None:
    """提取 {width, height} 与 obs_raw 列表。"""
    image_info = next((item for item in data if "Image" in item), None)
    obs_raw = next((item for item in data if "obs_raw" in item), None)
    if image_info is None or obs_raw is None:
        return None
    info = image_info["Image"]
    return {
        "width": int(info["width"]),
        "height": int(info["height"]),
        "obs_raw": obs_raw["obs_raw"],
    }


def _bbox_annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    rect: dict,
    keypoints: list | None = None,
    num_keypoints: int | None = None,
) -> dict:
    bbox = rect_to_bbox(rect)
    area = abs(bbox[2] * bbox[3])
    ann = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rect_to_segmentation(rect),
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }
    if keypoints is not None:
        ann["keypoints"] = keypoints
        ann["num_keypoints"] = num_keypoints if num_keypoints is not None else sum(1 for k in keypoints if k > 0) // 3
    return ann


def _extract_keypoints(obs: dict, keypoint_types: list) -> list:
    """按 point_type 索引生成 COCO keypoints 数组。"""
    kp = [0.0] * (3 * len(keypoint_types))
    for point in obs.get("key_point", []):
        try:
            index = int(point["point_type"]) * 3
            if index + 2 < len(kp):
                kp[index] = float(point["x"])
                kp[index + 1] = float(point["y"])
                kp[index + 2] = 2 if float(point["point_conf"]) > 0 else 0
        except (KeyError, TypeError, ValueError):
            continue
    return kp


def convert_to_coco(
    json_dir: str,
    output_dir: str,
    output_file_name: str,
    variant: str = "standard",
) -> dict:
    """转换原始标注为 COCO json。

    variant:
      - standard : 3 类（car/cyc/ped），无关键点（原 convert_to_coco.py）
      - mini     : 4 类，无关键点（原 mini_coco_created.py）
      - keypoints: 4 类 + fullcar/cyclist 前/后关键点（原 to_coco.py）
      - point-box: 关键点转为小检测框，6 类（原 to_coco_point_to_box_test.py）
    """
    if variant == "standard":
        obs_map = {k: v for k, v in OBS_TYPE_TO_CATEGORY.items() if v <= 3}
    else:
        obs_map = dict(OBS_TYPE_TO_CATEGORY)

    if variant == "point-box":
        obs_map.update(KEYPOINT_CATEGORIES)
    with_keypoints = variant in ("keypoints",)
    categories = _build_categories(obs_map, with_keypoints=with_keypoints)

    images = []
    annotations = []
    image_id = 1
    annotation_id = 1
    skipped = 0

    for json_file in iter_json_files(json_dir):
        try:
            data = load_json(os.path.join(json_dir, json_file))
            parsed = _parse_image(data)
            if parsed is None:
                skipped += 1
                continue
            images.append(
                {
                    "id": image_id,
                    "width": parsed["width"],
                    "height": parsed["height"],
                    "file_name": json_file[:-5] + ".jpg",
                }
            )
            for obs in parsed["obs_raw"]:
                obs_type = obs.get("type")
                if obs_type not in obs_map:
                    continue
                category_id = obs_map[obs_type]

                if variant == "point-box" and "key_point" in obs:
                    # 关键点转小检测框（原版实验逻辑）
                    for point in obs["key_point"][:2]:
                        cat = obs_map["ObstacleRawModel_front"] if point is obs["key_point"][0] else obs_map["ObstacleRawModel_rear"]
                        x, y = float(point["x"]), float(point["y"])
                        bbox = [float(obs["Rect"]["left"]), float(obs["Rect"]["rtop"]), x - float(obs["Rect"]["left"]), y - float(obs["Rect"]["rtop"])]
                        annotations.append(
                            {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": cat,
                                "segmentation": [float(obs["Rect"]["left"]), float(obs["Rect"]["rtop"]), x, y],
                                "area": 1.0,
                                "bbox": bbox,
                                "iscrowd": 0,
                                "keypoints": [],
                                "num_keypoints": 0,
                            }
                        )
                        annotation_id += 1
                    continue

                keypoints = None
                if with_keypoints:
                    keypoint_types = ["front", "rear"] if _category_name(obs_type) in ("fullcar", "cyclist") else []
                    keypoints = _extract_keypoints(obs, keypoint_types)
                annotations.append(
                    _bbox_annotation(
                        annotation_id, image_id, category_id, obs["Rect"], keypoints=keypoints
                    )
                )
                annotation_id += 1
            image_id += 1
        except (ValueError, KeyError, TypeError) as exc:
            print(f"跳过 {json_file}: {exc}")
            skipped += 1

    coco_data = {"images": images, "annotations": annotations, "categories": categories}
    output_path = os.path.join(output_dir, output_file_name)
    save_json(coco_data, output_path)
    print(f"转换完成：{output_path}，图片 {len(images)}，标注 {len(annotations)}，跳过 {skipped}")
    return {"images": len(images), "annotations": len(annotations), "skipped": skipped}
