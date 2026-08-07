"""通用文件与数据工具。"""

import json
import os
import shutil
from typing import Iterable, Iterator


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def iter_json_files(folder: str) -> Iterator[str]:
    """遍历目录中的 json 文件（按名称排序）。"""
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".json"):
            yield name


def json_path_for(image_name: str) -> str:
    return image_name[:-4] + ".json" if image_name.lower().endswith(".jpg") else image_name


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_copy(src: str, dst: str) -> bool:
    """复制文件，源不存在时返回 False。"""
    if not os.path.isfile(src):
        return False
    ensure_dir(os.path.dirname(dst) or ".")
    shutil.copy(src, dst)
    return True


def group_images_by_timestamp(
    image_files: Iterable[str], max_time_diff_seconds: int
) -> list[list[str]]:
    """按文件名前 14 位时间戳分组：时间差在阈值内的图片归为一组。"""
    files = sorted(image_files, key=lambda f: int(f[:14]))
    groups: list[list[str]] = []
    current: list[str] = []
    prev_ts: int | None = None
    for image_file in files:
        ts = int(image_file[:14])
        if prev_ts is None or ts - prev_ts <= max_time_diff_seconds:
            current.append(image_file)
        else:
            groups.append(current)
            current = [image_file]
        prev_ts = ts
    if current:
        groups.append(current)
    return groups


def rect_to_bbox(rect: dict) -> list[float]:
    """Rect {left, rtop, right, bottom} → COCO bbox [x, y, w, h]。"""
    left = float(rect["left"])
    rtop = float(rect["rtop"])
    right = float(rect["right"])
    bottom = float(rect["bottom"])
    return [left, rtop, right - left, bottom - rtop]


def rect_to_segmentation(rect: dict) -> list[list[float]]:
    left = float(rect["left"])
    rtop = float(rect["rtop"])
    right = float(rect["right"])
    bottom = float(rect["bottom"])
    return [[left, rtop, right, bottom]]
