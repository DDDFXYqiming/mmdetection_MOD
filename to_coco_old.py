import os
import json
import random
import shutil

def split_dataset(json_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    print('trying to split the data set...')
    image_files = [f for f in os.listdir(json_dir) if f.endswith(".jpg")]
    total_images = len(image_files)

    random.shuffle(image_files)

    # 计算分割图片数量
    num_train = int(total_images * train_ratio)
    num_val = int(total_images * val_ratio)

    print('data for train:{},val:{},test:{}'.format(num_train,num_val,(total_images-num_val-num_train)))

    train_dir = os.path.join(output_dir, "coco", "train2017")
    val_dir = os.path.join(output_dir, "coco", "val2017")
    test_dir = os.path.join(output_dir, "coco", "test2017")

    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 同时复制图片和json文件
    for i, image_file in enumerate(image_files):
        try:
            json_file = image_file[:-4] + ".json"
            src_image_path = os.path.join(json_dir, image_file)
            src_json_path = os.path.join(json_dir, json_file)

            if i < num_train:
                dst_image_path = os.path.join(train_dir, image_file)
                dst_json_path = os.path.join(train_dir, json_file)
            elif i < num_train + num_val:
                dst_image_path = os.path.join(val_dir, image_file)
                dst_json_path = os.path.join(val_dir, json_file)
            else:
                dst_image_path = os.path.join(test_dir, image_file)
                dst_json_path = os.path.join(test_dir, json_file)

            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(src_json_path, dst_json_path)
        except:
            continue

def convert_to_coco_format(json_dir, output_dir, output_file_name):
    images = []
    annotations = []
    categories = []
    image_id = 1
    annotation_id = 1

    # 定义物体类型到类别ID的映射
    obs_type_to_category = {
        "ObstacleRawModel_FullCar": 1,
        "ObstacleRawModel_Cyclist": 2,
        "ObstacleRawModel_Ped": 3,
        "ObstacleRawModel_Car": 4
    }

    # 自动生成categories列表
    for obs_type, category_id in obs_type_to_category.items():
        category_name = obs_type.lower().replace("obstaclerawmodel_", "")
        categories.append({
            "id": category_id,
            "name": category_name,
            "supercategory": "Vehicle"
        })

    for json_file in os.listdir(json_dir):
        # print(json_file)
        if json_file.endswith(".json"):
            print(json_file)
            with open(os.path.join(json_dir, json_file), "r") as f:

                data = json.load(f)
            
            # print(data)
            try:
                image_info = next(item for item in data if "Image" in item)
                # print(image_info)
                image_info = image_info["Image"]
                image_width = image_info["width"]
                image_height = image_info["height"]
                file_name = json_file.replace(".json", ".jpg")

                image_data = {
                    "id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": file_name
                }
                images.append(image_data)

                obs_raw = next(item for item in data if "obs_raw" in item)
                obs_raw = obs_raw["obs_raw"]
                for obs in obs_raw:
                    obs_type = obs["type"]
                    if obs_type in obs_type_to_category:
                        # 使用障碍物类型对应的类别ID
                        category_id = obs_type_to_category[obs_type]
                        # coco 坐标格式（X，Y，W，H）其中X,Y是左上角的坐标
                        bbox = [obs["Rect"]["left"], obs["Rect"]["rtop"], obs["Rect"]["right"] - obs["Rect"]["left"],
                                obs["Rect"]["bottom"] - obs["Rect"]["rtop"]]
                        area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                        # area = 1.0
                        segmentation = [[obs["Rect"]["left"], obs["Rect"]["rtop"], obs["Rect"]["right"], obs["Rect"]["bottom"]]]

                        annotation_data = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        }
                        annotations.append(annotation_data)
                        annotation_id += 1
                    else:
                        # 未知障碍物类型
                        pass

                image_id += 1
                
            except StopIteration:
                print(f"Error {json_file}：找不到所需信息。")
    
    # categories = [{
    #     "id": 1,
    #     "name": "car",
    #     "supercategory": "Vehicle"
    #     },{
    #     "id": 2,
    #     "name": "cyc",
    #     "supercategory": "Vehicle"
    # },{
    #     "id": 3,
    #     "name": "ped",
    #     "supercategory": "Vehicle"
    # }]

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, output_file_name)

    with open(output_file, "w") as f:
        json.dump(coco_data, f)

def delete_json_files_in_coco_subdirs(coco_dir):
    subdirs = ["train2017", "test2017", "val2017"]

    for subdir in subdirs:
        subdir_path = os.path.join(coco_dir, subdir)
        if os.path.exists(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".json"):
                    file_path = os.path.join(subdir_path, file)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
        
if __name__ == "__main__":
    # json_dir = "./J2camera"
    json_dir = "./J2camera_mini_selected_23"
    # json_dir = "./J2camera_selected_50"
    output_dir = "./mmdetection-main/data"
    # split_dataset(json_dir, output_dir)
    # split_dataset(json_dir, output_dir, train_ratio=0.98, val_ratio=0.01, test_ratio=0.01)
    split_dataset(json_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    output_file_name_list=["instances_train2017.json","instances_val2017.json","instances_test2017.json"]
    json_dir_list=["./mmdetection-main/data/coco/train2017","./mmdetection-main/data/coco/val2017","./mmdetection-main/data/coco/test2017"]

    output_dir = "./mmdetection-main/data/coco/annotations"
    for i in range(len(json_dir_list)):
        convert_to_coco_format(json_dir_list[i], output_dir, output_file_name_list[i])

    coco_dir = "./mmdetection-main/data/coco"
    delete_json_files_in_coco_subdirs(coco_dir)
