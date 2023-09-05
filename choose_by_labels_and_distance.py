import os
import json
import shutil
import math

# 选定的物体类型列表
choosen_list = ["ObstacleRawModel_Cyclist", "ObstacleRawModel_Ped", "ObstacleRawModel_FullCar"]

# 选定的距离阈值
choosen_distance = {"ObstacleRawModel_Cyclist": 10000, "ObstacleRawModel_Ped": 10000, "ObstacleRawModel_FullCar": 10000}

# 每隔多少个数据处理一次
interval = 4

# 目标距离
target_distance = 50

# 初始化计数器
origin, selected = 0, 0

# 计算物体距离的函数
def calculate_distance(obj_type, rect_info):
    # 假设摄像头参数
    camera_height = 1.5  # 摄像头高度（单位：米）
    camera_pitch = 0.0   # 摄像头俯仰角（单位：弧度）
    camera_focal_length = 1449.0  # 摄像头焦距（单位：像素）

    # 解析物体框的坐标信息
    bottom = rect_info["bottom"]
    left = rect_info["left"]
    right = rect_info["right"]
    rtop = rect_info["rtop"]

    # 计算物体在图像上的高度（单位：像素）
    obj_height_pixels = rtop - bottom

    # 根据不同物体类型计算距离
    if obj_type == "ObstacleRawModel_Cyclist" or obj_type == "ObstacleRawModel_Ped":
        # 假设行人和骑车者的平均身高为1.7米
        obj_height = 1.7
        distance = camera_height * obj_height_pixels / (obj_height * 2 * obj_height_pixels * (1 / camera_focal_length))
    elif obj_type == "ObstacleRawModel_FullCar":
        # 假设汽车的平均长度为4.5米
        obj_length = 4.5
        distance = camera_height * obj_length * camera_focal_length / (right - left)

    return distance

# 检查是否存在符合条件的目标
def check_for_target(json_path):
    with open(json_path, 'r', errors='ignore') as json_file:
        try:
            data = json.load(json_file)
            for item in data:
                if "obs_raw" in item and isinstance(item["obs_raw"], list):
                    for obs_data in item["obs_raw"]:
                        if "type" in obs_data and obs_data["type"] in choosen_list:
                            react = obs_data.get("Rect", {})
                            distance = calculate_distance(obs_data["type"], react)
                            if distance <= target_distance:
                                return True
        except:
            return False
    return False

# 主函数
def main():
    # 设置原始数据文件夹路径和目标数据文件夹路径
    origin_folder = "/home/omnisky/disk14/team/data/J2camera"
    output_folder = "./J2camera_selected_{}".format(target_distance)
    
    # 如果目标数据文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历原始数据文件夹中的所有文件
    for filename in os.listdir(origin_folder):
        if filename.endswith(".json"):
            global origin, selected
            origin += 1
            if origin % interval != 0:
                continue
            json_path = os.path.join(origin_folder, filename)
            image_path = os.path.join(origin_folder, filename.replace(".json", ".jpg"))

            # 检查json文件中是否包含符合条件的目标数据
            if check_for_target(json_path):
                try:
                    # 复制图片到目标数据文件夹
                    shutil.copy(image_path, os.path.join(output_folder, filename.replace(".json", ".jpg")))
                    shutil.copy(json_path, os.path.join(output_folder, filename))
                    selected += 1
                    print(f"已保存图像 {filename.replace('.json', '.jpg')} 及其对应的 JSON 文件。")
                except:
                    continue

    print("数据筛选完成。")
    print("{} 个数据被选中，总共处理了 {} 个数据。".format(selected, origin))

if __name__ == "__main__":
    main()
