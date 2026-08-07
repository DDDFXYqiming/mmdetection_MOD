"""COCO 标注可视化（原 visible.py / visible_old.py）。"""

import os

import cv2

from ..utils import ensure_dir, load_json


def visualize_annotations(
    annotation_file: str,
    image_folder: str,
    output_folder: str = "./annotated_images",
    draw_keypoints: bool = True,
) -> int:
    data = load_json(annotation_file)
    categories = {c["id"]: c["name"] for c in data["categories"]}
    annotations_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    ensure_dir(output_folder)
    saved = 0
    for image_info in data["images"]:
        image_path = os.path.join(image_folder, image_info["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue
        for ann in annotations_by_image.get(image_info["id"], []):
            bbox = [int(c) for c in ann["bbox"]]
            name = categories.get(ann["category_id"], "?")
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if draw_keypoints:
                kps = ann.get("keypoints", [])
                for i in range(len(kps) // 3):
                    x, y, visible = int(kps[i * 3]), int(kps[i * 3 + 1]), int(kps[i * 3 + 2])
                    color = (0, 0, 255) if visible == 2 else (255, 0, 0)
                    cv2.circle(image, (x, y), 3, color, -1)
        cv2.imwrite(os.path.join(output_folder, image_info["file_name"]), image)
        saved += 1
    print(f"可视化完成：{saved} 张图片 -> {output_folder}")
    return saved
