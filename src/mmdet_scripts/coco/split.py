"""数据集分割：按时间戳分组后分配到 train/val/test，或按批次分片。"""

import os
import shutil

from ..utils import ensure_dir, group_images_by_timestamp, json_path_for, safe_copy


def split_dataset(
    json_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    max_time_diff_minutes: float = 60,
    sort_groups_by_size: bool = True,
) -> dict[str, int]:
    """将 json_dir 中的 jpg+json 对按时间戳分组后分配到 coco/train2017 等目录。

    与原版行为一致：整组图片进入同一集合，避免连续帧被拆散。
    """
    print(f"分割数据集：{json_dir} -> {output_dir}")
    image_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".jpg")]
    total = len(image_files)
    print(f"总图片数量: {total}")
    if total == 0:
        return {"train": 0, "val": 0, "test": 0}

    max_diff = int(max_time_diff_minutes * 60)
    groups = group_images_by_timestamp(image_files, max_diff)
    if sort_groups_by_size:
        groups.sort(key=len, reverse=True)

    num_train = int(total * train_ratio)
    num_val = int(total * val_ratio)
    print(f"预估训练集数量: {num_train}, 验证集数量: {num_val}, 测试集数量: {total - num_train - num_val}")

    split_dirs = {
        "train": os.path.join(output_dir, "coco", "train2017"),
        "val": os.path.join(output_dir, "coco", "val2017"),
        "test": os.path.join(output_dir, "coco", "test2017"),
    }
    for d in split_dirs.values():
        ensure_dir(d)

    counts = {"train": 0, "val": 0, "test": 0}
    remaining_train, remaining_val = num_train, num_val
    for group in groups:
        if remaining_train > 0:
            target = "train"
            remaining_train -= len(group)
        elif remaining_val > 0:
            target = "val"
            remaining_val -= len(group)
        else:
            target = "test"
        for image_file in group:
            json_file = json_path_for(image_file)
            safe_copy(os.path.join(json_dir, image_file), os.path.join(split_dirs[target], image_file))
            safe_copy(os.path.join(json_dir, json_file), os.path.join(split_dirs[target], json_file))
            counts[target] += 1

    print(f"实际分配训练集数量: {counts['train']}, 验证集数量: {counts['val']}, 测试集数量: {counts['test']}")
    return counts


def batch_split(json_dir: str, output_dir: str, batch_size: int = 500) -> dict[str, int]:
    """将 json+图片对按固定批次复制到 batch_N 子目录。"""
    json_files = sorted(f for f in os.listdir(json_dir) if f.lower().endswith(".json"))
    total = len(json_files)
    batch_count = (total + batch_size - 1) // batch_size
    print(f"总 JSON 文件数量: {total}，批次数量: {batch_count}")

    for i in range(batch_count):
        batch = json_files[i * batch_size : (i + 1) * batch_size]
        batch_dir = ensure_dir(os.path.join(output_dir, f"batch_{i + 1}"))
        for json_file in batch:
            image_file = json_file[:-5] + ".jpg"
            safe_copy(os.path.join(json_dir, json_file), os.path.join(batch_dir, json_file))
            safe_copy(os.path.join(json_dir, image_file), os.path.join(batch_dir, image_file))
        print(f"复制批次 {i + 1}：{len(batch)} 个 JSON 文件")
    return {"total": total, "batches": batch_count}
