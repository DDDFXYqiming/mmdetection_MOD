import os
import json

def convert_to_coco_format(json_dir, output_dir, output_file_name):
    images = []
    annotations = []
    categories = []
    image_id = 1
    annotation_id = 1

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
                    if obs["type"] == "ObstacleRawModel_FullCar":
                        # coco 坐标格式（X，Y，W，H）其中X,Y是左上角的坐标
                        bbox = [obs["Rect"]["left"], obs["Rect"]["rtop"], obs["Rect"]["right"] - obs["Rect"]["left"],
                                obs["Rect"]["bottom"] - obs["Rect"]["rtop"]]
                        # area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        area = 1.0

                        segmentation = [[obs["Rect"]["left"], obs["Rect"]["rtop"],obs["Rect"]["right"], obs["Rect"]["bottom"]]]

                        annotation_data = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        }
                        annotations.append(annotation_data)
                        annotation_id += 1
                    elif obs["type"] == "ObstacleRawModel_Cyclist":
                        # coco 坐标格式（X，Y，W，H）其中X,Y是左上角的坐标
                        bbox = [obs["Rect"]["left"], obs["Rect"]["rtop"], obs["Rect"]["right"] - obs["Rect"]["left"],
                                obs["Rect"]["bottom"] - obs["Rect"]["rtop"]]
                        # area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        area = 1.0

                        segmentation = [[obs["Rect"]["left"], obs["Rect"]["rtop"],obs["Rect"]["right"], obs["Rect"]["bottom"]]]

                        annotation_data = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 2,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        }
                        annotations.append(annotation_data)
                        annotation_id += 1
                    elif obs["type"] == "ObstacleRawModel_Ped":
                        # coco 坐标格式（X，Y，W，H）其中X,Y是左上角的坐标
                        bbox = [obs["Rect"]["left"], obs["Rect"]["rtop"], obs["Rect"]["right"] - obs["Rect"]["left"],
                                obs["Rect"]["bottom"] - obs["Rect"]["rtop"]]
                        # area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        area = 1.0

                        segmentation = [[obs["Rect"]["left"], obs["Rect"]["rtop"],obs["Rect"]["right"], obs["Rect"]["bottom"]]]

                        annotation_data = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 3,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        }
                        annotations.append(annotation_data)
                        annotation_id += 1

                image_id += 1
                
            except StopIteration:
                print(f"Error {json_file}：找不到所需信息。")
    
    categories = [{
        "id": 1,
        "name": "car",
        "supercategory": "Vehicle"
    },{
        "id": 2,
        "name": "cyc",
        "supercategory": "Vehicle"
    },{
        "id": 3,
        "name": "ped",
        "supercategory": "Vehicle"
    }]

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

if __name__ == "__main__":
    # # output_file = "instances_train2017.json"
    # # output_file = "instances_val2017.json"
    # output_file_name = "instances_test2017.json"
    # # json_dir = "./mmdetection-main/data/coco/train2017"
    # # json_dir = "./mmdetection-main/data/coco/val2017"
    # json_dir = "./mmdetection-main/data/coco/test2017"

    # output_dir = "./mmdetection-main/data/coco/annotations"
    # convert_to_coco_format(json_dir, output_dir, output_file_name)

    output_file_name_list=["instances_train2017.json","instances_val2017.json","instances_test2017.json"]
    json_dir_list=["./mmdetection-main/data/coco/train2017","./mmdetection-main/data/coco/val2017","./mmdetection-main/data/coco/test2017"]

    output_dir = "./mmdetection-main/data/coco/annotations"
    for i in range(len(json_dir_list)):
        convert_to_coco_format(json_dir_list[i], output_dir, output_file_name_list[i])