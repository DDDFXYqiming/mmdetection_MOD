    #   {
    #     "Rect": {
    #       "bottom": 692.0,
    #       "left": 1230.0,
    #       "right": 1777.5,
    #       "rtop": 285.5
    #     },
    #     "conf": 1.0,
    #     "key_point": [
    #       {
    #         "point_conf": 9.744873046875,
    #         "point_type": 0.0,
    #         "type": "ObstacleRawModel_FullCar",
    #         "x": 1420.2943115234375,
    #         "y": 714.4404907226563,
    #         "z": 0.0
    #       },
    #       {
    #         "point_conf": 9.9249267578125,
    #         "point_type": 1.0,
    #         "type": "ObstacleRawModel_FullCar",
    #         "x": 1259.6783447265625,
    #         "y": 569.1322631835938,
    #         "z": 0.0
    #       }
    #     ],
    #     "model": 1,
    #     "prop": [
    #       {
    #         "conf": 69.375,
    #         "name": "Small_Medium_Car",
    #         "property": 1,
    #         "type": 0
    #       },
    #       {
    #         "conf": 34.375,
    #         "name": "occluded",
    #         "property": 1,
    #         "type": 9
    #       }
    #     ],
    #     "type": "ObstacleRawModel_FullCar"
    #   },
    #   {
    #     "Rect": {
    #       "bottom": 363.5,
    #       "left": 850.5,
    #       "right": 925.25,
    #       "rtop": 298.75
    #     },
    #     "conf": 1.0,
    #     "model": 1,
    #     "type": "ObstacleRawModel_FullCar"
    #   },

import os
import json
import random
import shutil

def split_dataset(json_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, max_time_diff_minutes=60):
    print('分割数据集...')
    image_files = [f for f in os.listdir(json_dir) if f.endswith(".jpg")]
    total_images = len(image_files)

    # 根据时间戳对图像进行排序
    image_files.sort(key=lambda f: int(f[:14]))  # 假设前14个字符代表时间戳

    # 计算允许的最大时间差（单位：秒）
    max_time_diff_seconds = max_time_diff_minutes * 60

    print('总图片数量: {}'.format(total_images))

    train_dir = os.path.join(output_dir, "coco", "train2017")
    val_dir = os.path.join(output_dir, "coco", "val2017")
    test_dir = os.path.join(output_dir, "coco", "test2017")

    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 将图片按照时间戳分组，确保在最大允许时间差内的图片被放入同一组
    image_groups = []
    current_group = []
    for image_file in image_files:
        timestamp = int(image_file[:14])
        if not current_group or timestamp - current_group[-1][1] <= max_time_diff_seconds:
            current_group.append((image_file, timestamp))
        else:
            image_groups.append(current_group)
            current_group = [(image_file, timestamp)]
    if current_group:
        image_groups.append(current_group)

    sorted_image_groups = sorted(image_groups, key=lambda group: len(group), reverse=True)

    # 计算分割图片数量
    # print(image_groups)
    num_train = int(total_images*train_ratio)
    num_val = int(total_images*val_ratio)
    real_train = 0
    real_val = 0

    print('预估训练集数量: {}, 验证集数量: {}, 测试集数量: {}'.format(num_train, num_val, (total_images - num_val - num_train)))

    # 根据组的数量将图片分配到对应的集合（训练集、验证集、测试集）中
    for i, group in enumerate(sorted_image_groups):
        print("当前组图片数："+str(len(sorted_image_groups[i])))

        if num_train>0:
            for image_file, _ in group:
                json_file = image_file[:-4] + ".json"
                src_image_path = os.path.join(json_dir, image_file)
                src_json_path = os.path.join(json_dir, json_file)

                dst_image_path = os.path.join(train_dir, image_file)
                dst_json_path = os.path.join(train_dir, json_file)

                shutil.copy(src_image_path, dst_image_path)
                shutil.copy(src_json_path, dst_json_path)

                num_train -= 1
                real_train += 1
                # print(num_train)

        elif num_val>0:
            for image_file, _ in group:
                json_file = image_file[:-4] + ".json"
                src_image_path = os.path.join(json_dir, image_file)
                src_json_path = os.path.join(json_dir, json_file)

                dst_image_path = os.path.join(val_dir, image_file)
                dst_json_path = os.path.join(val_dir, json_file)

                shutil.copy(src_image_path, dst_image_path)
                shutil.copy(src_json_path, dst_json_path)

                num_val -= 1
                real_val += 1
                # print(num_val)

        else:
            for image_file, _ in group:
                json_file = image_file[:-4] + ".json"
                src_image_path = os.path.join(json_dir, image_file)
                src_json_path = os.path.join(json_dir, json_file)
                
                dst_image_path = os.path.join(test_dir, image_file)
                dst_json_path = os.path.join(test_dir, json_file)

                shutil.copy(src_image_path, dst_image_path)
                shutil.copy(src_json_path, dst_json_path)

    print('实际分配训练集数量: {}, 验证集数量: {}, 测试集数量: {}'.format(real_train, real_val, (total_images - real_val - real_train)))
    

def convert_to_coco_format(json_dir, output_dir, output_file_name):
    images = []
    annotations = []
    categories = []
    # 不再处理关键点数据，将 keypoints 设为 []
    keypoints = []
    image_id = 1
    annotation_id = 1

    # 定义障碍物类型到类别ID的映射，包括关键点类型
    obs_type_to_category = {
        "ObstacleRawModel_FullCar": 1,
        "ObstacleRawModel_Cyclist": 2,
        "ObstacleRawModel_Ped": 3,
        "ObstacleRawModel_Car": 4,
        "ObstacleRawModel_front": 5,
        "ObstacleRawModel_rear": 6,
    }

    # 自动生成categories列表，包括关键点类型的信息
    for obs_type, category_id in obs_type_to_category.items():
        category_name = obs_type.lower().replace("obstaclerawmodel_", "")
        category_keypoints = []
        skeleton = []
            
        categories.append({
            "id": category_id,
            "name": category_name,
            "supercategory": "Vehicle",
            "keypoints": category_keypoints,
            "skeleton": skeleton
        })

    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            print(json_file)
            with open(os.path.join(json_dir, json_file), "r") as f:
                data = json.load(f)

            try:
                image_info = next(item for item in data if "Image" in item)
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
                    if "key_point" in obs:
                        # 处理关键点数据，将其作为特别小的检测框添加到数据中
                        obs_type_front = "ObstacleRawModel_front"
                        obs_type_rear = "ObstacleRawModel_rear"

                        category_id = obs_type_to_category[obs_type_front]

                        x, y = obs["key_point"][0]["x"], obs["key_point"][0]["y"]
                        # coco 坐标格式（X，Y，W，H）其中X,Y是左上角的坐标
                        # 将关键点视为检测框
                        bbox = [obs["Rect"]["left"], obs["Rect"]["rtop"], x - obs["Rect"]["left"],
                                y - obs["Rect"]["rtop"]]
                        # bbox = [x, y, 7, 7]
                        area = 1
                        segmentation = [obs["Rect"]["left"], obs["Rect"]["rtop"], x, y]
                        # segmentation = [[x, y, x + 7, y + 7]]

                        annotation_data = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "keypoints": keypoints,
                            "num_keypoints": 0  # 关键点数量为0
                        }
                        annotations.append(annotation_data)
                        annotation_id += 1

                        
                        category_id = obs_type_to_category[obs_type_rear]
                        x, y = obs["key_point"][1]["x"], obs["key_point"][1]["y"]
                        # coco 坐标格式（X，Y，W，H）其中X,Y是左上角的坐标
                        # 将关键点视为检测框
                        bbox = [obs["Rect"]["left"], obs["Rect"]["rtop"], x - obs["Rect"]["left"],
                                y - obs["Rect"]["rtop"]]
                        # bbox = [x, y, 7, 7]
                        area = 1
                        segmentation = [obs["Rect"]["left"], obs["Rect"]["rtop"], x, y]
                        # segmentation = [[x, y, x + 7, y + 7]]

                        annotation_data = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "keypoints": keypoints,
                            "num_keypoints": 0  # 关键点数量为0
                        }
                        annotations.append(annotation_data)
                        annotation_id += 1
                        
                    if obs_type in obs_type_to_category and obs_type_to_category[obs_type] != 5 and obs_type_to_category[obs_type] != 6:
                        if obs_type_to_category[obs_type] != 5:
                            # 使用障碍物类型对应的类别ID
                            category_id = obs_type_to_category[obs_type]
                            # coco 坐标格式（X，Y，W，H）其中X,Y是左上角的坐标
                            bbox = [obs["Rect"]["left"], obs["Rect"]["rtop"], obs["Rect"]["right"] - obs["Rect"]["left"],
                                    obs["Rect"]["bottom"] - obs["Rect"]["rtop"]]
                            area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                            segmentation = [[obs["Rect"]["left"], obs["Rect"]["rtop"], obs["Rect"]["right"], obs["Rect"]["bottom"]]]

                            annotation_data = {
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": category_id,
                                "segmentation": segmentation,
                                "area": area,
                                "bbox": bbox,
                                "iscrowd": 0,
                                "keypoints": keypoints,
                                "num_keypoints": 0
                            }
                            annotations.append(annotation_data)
                            annotation_id += 1
                    else:
                        # 未知障碍物类型
                        pass

                image_id += 1
                
            except StopIteration:
                print(f"Error {json_file}：找不到所需信息。")

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
    # json_dir = "./J2camera_mini_selected_23"
    json_dir = "./J2camera_selected_23"
    # json_dir = "./J2camera_selected_50"
    output_dir = "./mmpose-main/data"
    # split_dataset(json_dir, output_dir)
    # split_dataset(json_dir, output_dir, train_ratio=0.98, val_ratio=0.01, test_ratio=0.01)
    split_dataset(json_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    output_file_name_list=["instances_train2017.json","instances_val2017.json","instances_test2017.json"]
    json_dir_list=["./mmpose-main/data/coco/train2017","./mmpose-main/data/coco/val2017","./mmpose-main/data/coco/test2017"]

    output_dir = "./mmpose-main/data/coco/annotations"
    for i in range(len(json_dir_list)):
        convert_to_coco_format(json_dir_list[i], output_dir, output_file_name_list[i])

    coco_dir = "./mmpose-main/data/coco"
    delete_json_files_in_coco_subdirs(coco_dir)
