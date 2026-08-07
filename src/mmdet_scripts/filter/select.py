"""按目标类型与距离阈值筛选数据（原 choose_by_labels_and_distance.py）。

距离估算：小孔成像模型 distance = 物体实际尺寸 * 焦距 / 像素尺寸。
原版行人公式（height_pixels 在分子分母同时出现）化简后与像素高度无关，
属明显笔误，本实现默认使用修正公式；可通过 --formula legacy 复现原行为。
"""

import json
import math
import os

from ..config import DEFAULT_CAMERA
from ..utils import ensure_dir, safe_copy


def calculate_distance(
    obj_type: str,
    rect: dict,
    formula: str = "corrected",
    camera: dict | None = None,
) -> float:
    cam = {**DEFAULT_CAMERA, **(camera or {})}
    bottom = float(rect["bottom"])
    rtop = float(rect["rtop"])
    left = float(rect["left"])
    right = float(rect["right"])
    height_px = abs(rtop - bottom)
    width_px = abs(right - left)

    if obj_type in ("ObstacleRawModel_Cyclist", "ObstacleRawModel_Ped"):
        obj_height = cam["obj_height"]
        if formula == "legacy":
            # 原版公式（笔误）：分子分母同时含 height_px，化简为与图像高度无关
            return cam["height"] * height_px / (obj_height * 2 * height_px * (1 / cam["focal_length"]))
        if height_px <= 0:
            return math.inf
        return (obj_height * cam["focal_length"]) / height_px
    if obj_type == "ObstacleRawModel_FullCar":
        obj_length = cam["car_length"]
        if width_px <= 0:
            return math.inf
        return (obj_length * cam["focal_length"]) / width_px
    return math.inf


def contains_target(
    json_path: str,
    target_types: list[str],
    max_distance: float,
    formula: str = "corrected",
    camera: dict | None = None,
) -> bool:
    try:
        with open(json_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    for item in data:
        obs_raw = item.get("obs_raw")
        if not isinstance(obs_raw, list):
            continue
        for obs in obs_raw:
            if obs.get("type") in target_types:
                distance = calculate_distance(obs["type"], obs.get("Rect", {}), formula, camera)
                if distance <= max_distance:
                    return True
    return False


def select_by_labels_and_distance(
    origin_folder: str,
    output_folder: str,
    target_types: list[str] | None = None,
    max_distance: float = 50,
    interval: int = 4,
    formula: str = "corrected",
) -> dict[str, int]:
    """每隔 interval 个 json 检查一次，命中目标类型且在距离阈值内则复制图片+json。"""
    types = target_types or [
        "ObstacleRawModel_Cyclist",
        "ObstacleRawModel_Ped",
        "ObstacleRawModel_FullCar",
    ]
    ensure_dir(output_folder)
    origin = 0
    selected = 0
    for filename in sorted(os.listdir(origin_folder)):
        if not filename.lower().endswith(".json"):
            continue
        origin += 1
        if origin % interval != 0:
            continue
        json_path = os.path.join(origin_folder, filename)
        image_path = os.path.join(origin_folder, filename[:-5] + ".jpg")
        if contains_target(json_path, types, max_distance, formula):
            ok_img = safe_copy(image_path, os.path.join(output_folder, filename[:-5] + ".jpg"))
            ok_json = safe_copy(json_path, os.path.join(output_folder, filename))
            if ok_img and ok_json:
                selected += 1
                print(f"已保存 {filename[:-5]}.jpg 及其 JSON")
    print(f"数据筛选完成：{selected} 个数据被选中，总共处理了 {origin} 个数据")
    return {"origin": origin, "selected": selected}
