import os
import json

json_folder = "./driving_scene"
json_files = [file for file in os.listdir(json_folder) if file.endswith(".json")]

deleted_count = 0
total_count = len(json_files)

# 遍历每个 JSON 文件
for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)
    image_file = json_file.replace(".json", ".jpg")
    image_path = os.path.join(json_folder, image_file)

    # 检查相应的图像文件是否存在
    if not os.path.isfile(image_path):
        try:
            os.remove(json_path)  # 删除没有对应图像的 JSON 文件
            deleted_count += 1
            print(f"已删除 {json_file}")
        except Exception as e:
            print(f"无法删除 {json_file}: {e}")

print(f"总 JSON 文件数: {total_count}")
print(f"已删除 JSON 文件数: {deleted_count}")
print(f"剩余 JSON 文件数: {total_count - deleted_count}")
