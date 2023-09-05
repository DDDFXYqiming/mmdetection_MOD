import cv2
import os
import json

# 标注文件与图片文件路径
# annotation_file = "./mmdetection-main/data/coco/annotations/instances_train2017.json"
annotation_file = "./mmdetection-main/data/coco/annotations/instances_val2017.json"
with open(annotation_file, "r") as f:
    data = json.load(f)

# image_folder = "./mmdetection-main/data/coco/train2017"
image_folder = "./mmdetection-main/data/coco/val2017"


categories_dict = {category["id"]: category["name"] for category in data["categories"]}

# 遍历标注文件可视化标注框，根据图片ID打包
annotations_by_image = {}
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    if image_id in annotations_by_image:
        annotations_by_image[image_id].append(annotation)
    else:
        annotations_by_image[image_id] = [annotation]

for image_info in data["images"]:
    image_id = image_info["id"]
    image_width = image_info["width"]
    image_height = image_info["height"]
    image_name = image_info["file_name"]
    image_path = os.path.join(image_folder, image_name)

    image = cv2.imread(image_path)

    if image_id in annotations_by_image:
        annotations = annotations_by_image[image_id]
        for annotation in annotations:
            bbox = annotation["bbox"]
            bbox = [int(coord) for coord in bbox]  # Convert to integers

            category_id = annotation["category_id"]
            category_name = categories_dict[category_id]

            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, category_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存可视化标注信息图片
    output_folder = "./annotated_images"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)

print("Annotation images saved successfully.")