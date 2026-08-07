"""数据集清理：删除无对应图片的 json、清理 COCO 子目录中的 json。"""

import os

from ..config import COCO_SPLITS


def delete_orphan_json(folder: str) -> dict[str, int]:
    """删除没有对应 jpg 的 json 文件。"""
    deleted = 0
    total = 0
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".json"):
            continue
        total += 1
        image = name[:-5] + ".jpg"
        if not os.path.isfile(os.path.join(folder, image)):
            os.remove(os.path.join(folder, name))
            deleted += 1
            print(f"已删除 {name}")
    print(f"总 JSON 文件数: {total}，已删除: {deleted}，剩余: {total - deleted}")
    return {"total": total, "deleted": deleted}


def delete_json_in_coco_subdirs(coco_dir: str) -> int:
    """删除 coco/train2017|val2017|test2017 下的 json（标注生成前的中间产物）。"""
    deleted = 0
    for subdir in COCO_SPLITS:
        subdir_path = os.path.join(coco_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file in os.listdir(subdir_path):
            if file.lower().endswith(".json"):
                os.remove(os.path.join(subdir_path, file))
                deleted += 1
    print(f"已删除 {deleted} 个 json 文件")
    return deleted
