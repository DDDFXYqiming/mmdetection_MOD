"""pkl 预测结果可视化（原 pkl_cv.py / combined_cv.py）。"""

import os
import pickle

import cv2

from ..config import PRED_LABEL_NAMES
from ..utils import ensure_dir, load_json


def _load_pkl(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def _draw_predictions(img, item, score_threshold: float, alpha: float) -> None:
    for j in range(len(item["pred_instances"]["bboxes"])):
        score = item["pred_instances"]["scores"][j]
        if float(score) <= score_threshold:
            continue
        bbox = item["pred_instances"]["bboxes"][j]
        label = int(item["pred_instances"]["labels"][j])
        overlay = img.copy()
        color = (255 - label * 30, label * 60, label * 80)
        cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        label_text = f"{PRED_LABEL_NAMES.get(label, label)}:{round(float(score) * 100, 1)}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_width, text_height = text_size[0], text_size[1]
        cv2.rectangle(
            overlay,
            (int(bbox[0] + 2), int(bbox[1]) - text_height + 28),
            (int(bbox[0]) + text_width, int(bbox[1]) + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            overlay,
            label_text,
            (int(bbox[0] + 2), int(bbox[1]) + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        if label in (0, 1):
            kp1 = item["pred_instances"]["keypoints1"][j]
            kp2 = item["pred_instances"]["keypoints2"][j]
            cv2.circle(img, (int(kp1[0]), int(kp1[1])), 3, (255, 0, 0), -1)
            cv2.circle(img, (int(kp2[0]), int(kp2[1])), 3, (0, 0, 255), -1)


def _draw_ground_truth(img, image_info, annotations, categories: dict) -> None:
    for ann in annotations:
        bbox = [int(c) for c in ann["bbox"]]
        name = categories.get(ann["category_id"], "?")
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.putText(img, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        kps = ann.get("keypoints", [])
        for i in range(len(kps) // 3):
            x, y, visible = int(kps[i * 3]), int(kps[i * 3 + 1]), int(kps[i * 3 + 2])
            color = (0, 0, 255) if visible == 2 else (255, 0, 0)
            cv2.circle(img, (x, y), 3, color, -1)


def visualize_pkl_predictions(
    pkl_path: str,
    output_dir: str = "./pkl_cv",
    score_threshold: float = 0.5,
    alpha: float = 0.65,
) -> int:
    """预测结果可视化（原 pkl_cv.py）。"""
    ensure_dir(output_dir)
    data = _load_pkl(pkl_path)
    saved = 0
    for i, item in enumerate(data):
        img = cv2.imread(item["img_path"])
        if img is None:
            print(f"跳过无法读取的图片 {item['img_path']}")
            continue
        _draw_predictions(img, item, score_threshold, alpha)
        output_path = os.path.join(output_dir, f"visualization_{i}.jpg")
        cv2.imwrite(output_path, img)
        saved += 1
    print(f"预测可视化完成：{saved} 张 -> {output_dir}")
    return saved


def visualize_pkl_compare(
    pkl_path: str,
    annotation_file: str,
    image_folder: str,
    output_dir: str = "./output_combined_images",
    score_threshold: float = 0.3,
    alpha: float = 0.6,
) -> int:
    """预测 vs 标注对比拼接（原 combined_cv.py）。"""
    ensure_dir(output_dir)
    data = _load_pkl(pkl_path)
    gt = load_json(annotation_file)
    categories = {c["id"]: c["name"] for c in gt["categories"]}
    images_info = {info["file_name"]: info for info in gt["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in gt["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    saved = 0
    for i, item in enumerate(data):
        img = cv2.imread(item["img_path"])
        if img is None:
            continue
        _draw_predictions(img, item, score_threshold, alpha)
        filename = item["img_path"].split("/")[-1]
        info = images_info.get(filename)
        orig = cv2.imread(os.path.join(image_folder, filename))
        if info is not None and orig is not None:
            _draw_ground_truth(orig, info, anns_by_image.get(info["id"], []), categories)
            combined = cv2.hconcat([orig, img])
        else:
            combined = img
        cv2.imwrite(os.path.join(output_dir, f"combined_visualization_{i}.jpg"), combined)
        saved += 1
    print(f"对比可视化完成：{saved} 张 -> {output_dir}")
    return saved
