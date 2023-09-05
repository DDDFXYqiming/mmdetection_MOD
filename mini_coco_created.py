import os
import json
import shutil
import random

def split_data(data_dir, val_ratio=0.2, test_ratio=0.1, categories_to_convert=['person']):
    # 如果不存在 "coco" 目录，则创建
    coco_dir = os.path.join(data_dir, "coco")
    if not os.path.exists(coco_dir):
        os.makedirs(coco_dir)
    
    # 如果不存在子目录（train2017、val2017、test2017），则创建
    for split in ["train2017", "val2017", "test2017"]:
        split_dir = os.path.join(coco_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

    # 如果不存在 "annotations" 目录，则创建
    annotations_dir = os.path.join(coco_dir, "annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    
    # 读取 instances_val2017.json 文件
    json_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # 获取图像和注释信息
    images = data["images"]
    annotations = data["annotations"]
    
    # 随机打乱图像列表，以便划分为训练、验证和测试集
    random.shuffle(images)
    total_count = len(images)
    val_count = int(val_ratio * total_count)
    test_count = int(test_ratio * total_count)
    
    # 划分图像为训练、验证和测试集
    train_images = images[val_count + test_count:]
    val_images = images[:val_count]
    test_images = images[val_count:val_count + test_count]
    
    # 将划分的图像保存到对应目录中
    for image in train_images:
        src_image_path = os.path.join(data_dir, "val2017", image["file_name"])
        dest_image_path = os.path.join(coco_dir, "train2017", image["file_name"])
        shutil.copyfile(src_image_path, dest_image_path)
        
    for image in val_images:
        src_image_path = os.path.join(data_dir, "val2017", image["file_name"])
        dest_image_path = os.path.join(coco_dir, "val2017", image["file_name"])
        shutil.copyfile(src_image_path, dest_image_path)
    
    for image in test_images:
        src_image_path = os.path.join(data_dir, "val2017", image["file_name"])
        dest_image_path = os.path.join(coco_dir, "test2017", image["file_name"])
        shutil.copyfile(src_image_path, dest_image_path)
    
    # 保存训练、验证和测试集的注释信息
    train_annotations = []
    val_annotations = []
    test_annotations = []
    done = 0
    image_id = 0
    for annotation in annotations:
        print('alive,done:{}'.format(done))
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]  # 获取注释的 category_id
        category_name = None
        for category in data["categories"]:
            if category["id"] == category_id:
                category_name = category["name"]
                break
        if category_name is None:
            raise ValueError(f"Category ID {category_id} not found in the data['categories'] list.")
        if any(image["id"] == image_id for image in train_images):
            # 仅保留特定类别的注释
            if category_name in categories_to_convert:
                train_annotations.append(annotation)
        elif any(image["id"] == image_id for image in val_images):
            # 仅保留特定类别的注释
            if category_name in categories_to_convert:
                val_annotations.append(annotation)
        elif any(image["id"] == image_id for image in test_images):
            # 仅保留特定类别的注释
            if category_name in categories_to_convert:
                test_annotations.append(annotation)
        done += 1
    
    train_json = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": [category for category in data["categories"] if category["name"] in categories_to_convert]
    }
    
    val_json = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": [category for category in data["categories"] if category["name"] in categories_to_convert]
    }
    
    test_json = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": [category for category in data["categories"] if category["name"] in categories_to_convert]
    }
    
    with open(os.path.join(coco_dir, "annotations", "instances_train2017.json"), "w") as f:
        json.dump(train_json, f)
    
    with open(os.path.join(coco_dir, "annotations", "instances_val2017.json"), "w") as f:
        json.dump(val_json, f)
    
    with open(os.path.join(coco_dir, "annotations", "instances_test2017.json"), "w") as f:
        json.dump(test_json, f)

if __name__ == "__main__":
    data_dir = "mmdetection-main\data\coco_l"
    
    categories_to_convert = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign']

    # categories_to_convert = ['person', 'bicycle', 'car']  # 只转换指定类别
    split_data(data_dir, categories_to_convert=categories_to_convert)
