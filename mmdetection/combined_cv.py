import os
import pickle
import cv2
import numpy as np
import torch
import json

pre_score = 0.3  # 得分阈值
alpha = 0.6  # 设置透明度

# 读取pkl文件内容
path = './results.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

# 标注文件与图片文件路径
annotation_file = "./data/coco/annotations/instances_val2017.json"
with open(annotation_file, "r") as f:
    data_annotations = json.load(f)

image_folder = "./data/coco/val2017"
categories_dict = {category["id"]: category["name"] for category in data_annotations["categories"]}

# 两个目录路径
directory1 = './annotated_images'
directory2 = './pkl_cv'

# 创建保存拼接结果的目录
output_dir = 'output_combined_images'
os.makedirs(output_dir, exist_ok=True)

# 遍历数据进行可视化和拼接
for i, item in enumerate(data):
    img_path = item['img_path']

    img = cv2.imread(img_path)

    for j in range(len(item['pred_instances']['bboxes'])):
        score = item['pred_instances']['scores'][j]
        bbox = item['pred_instances']['bboxes'][j]
        label = item['pred_instances']['labels'][j]
        keypoints1 = item['pred_instances']['keypoints1'][j]
        keypoints2 = item['pred_instances']['keypoints2'][j]

        if score <= pre_score:
            continue

        score = round(score.item() * 100.0, 1)

        # 创建一个与图像相同大小的空白图像作为透明图层
        overlay = img.copy()

        # 绘制边界框
        cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255 - int(label*30), int(label*60), int(label*80)), 2)

        label_name = {0: 'fullcar', 1: 'cyclist', 2: 'ped', 3: 'car'}

        cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255 - int(label*30), int(label*60), int(label*80)), 2)

        label_text = f'{label_name[int(label)]}:{score}'

        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_width, text_height = text_size[0], text_size[1]

        cv2.rectangle(overlay, (int(bbox[0] + 2), int(bbox[1]) - text_height + 28), (int(bbox[0]) + text_width, int(bbox[1]) + 2), (0, 0, 0), -1)

        cv2.putText(overlay, label_text, (int(bbox[0] + 2), int(bbox[1]) + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        if label == 0 or label == 1:
            x1, y1 = int(keypoints1[0]), int(keypoints1[1])
            x2, y2 = int(keypoints2[0]), int(keypoints2[1])
            cv2.circle(img, (x1, y1), 3, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1)

    # 保存可视化结果
    parts = img_path.split('/')
    last_part = parts[-1]
    output_path = os.path.join(directory2, last_part)
    cv2.imwrite(output_path, img)
    print(f'Saved visualization {i} at {output_path}')

    # 获取原标签图像的路径
    orig_image_info = [info for info in data_annotations['images'] if info['file_name'] == last_part][0]
    orig_image_name = orig_image_info['file_name']
    orig_image_path = os.path.join(image_folder, orig_image_name)
    orig_img = cv2.imread(orig_image_path)

    # 从标注数据中获取对应图片的标签信息
    annotations = [anno for anno in data_annotations['annotations'] if anno['image_id'] == orig_image_info['id']]

    for annotation in annotations:
        bbox = annotation["bbox"]
        bbox = [int(coord) for coord in bbox]  # Convert to integers

        category_id = annotation["category_id"]
        category_name = categories_dict[category_id]

        cv2.rectangle(orig_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        cv2.putText(orig_img, category_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        keypoints = annotation["keypoints"]
        num_keypoints = len(keypoints) // 3
        for k in range(num_keypoints):
            x = int(keypoints[k * 3])
            y = int(keypoints[k * 3 + 1])
            visible = int(keypoints[k * 3 + 2])

            if visible == 2:  # 可见
                cv2.circle(orig_img, (x, y), 3, (0, 0, 255), -1)
            elif visible == 1:  # 不可见
                cv2.circle(orig_img, (x, y), 3, (255, 0, 0), -1)

    # 将原标签图像和预测标签图像水平拼接
    combined_img = cv2.hconcat([orig_img, img])

    # 保存拼接后的图片
    combined_output_path = os.path.join(output_dir, f'combined_visualization_{i}.jpg')
    cv2.imwrite(combined_output_path, combined_img)
    print(f'Saved combined visualization {i} at {combined_output_path}')

print("可视化图片成功")
