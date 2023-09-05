import os
import shutil

json_folder = "J2camera_selected_50"
output_folder = "split_json2"

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有以 ".json" 结尾的文件列表
json_files = [file for file in os.listdir(json_folder) if file.endswith(".json")]

# 计算总共的文件数量和批次数量
total_count = len(json_files)
batch_size = 500
batch_count = (total_count + batch_size - 1) // batch_size

# 对每个批次进行处理
for i in range(batch_count):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, total_count)
    batch_json_files = json_files[start_idx:end_idx]

    batch_output_folder = os.path.join(output_folder, f"batch_{i + 1}")
    os.makedirs(batch_output_folder, exist_ok=True)

    # 复制每个批次的 JSON 文件和对应的图像文件
    for json_file in batch_json_files:
        json_path = os.path.join(json_folder, json_file)
        image_file = json_file.replace(".json", ".jpg")
        image_path = os.path.join(json_folder, image_file)
        if os.path.isfile(image_path):
            shutil.copy(json_path, batch_output_folder)
            shutil.copy(image_path, batch_output_folder)

    print(f"复制批次 {i + 1}：{len(batch_json_files)} 个 JSON 文件.")

print(f"总共的 JSON 文件数量：{total_count}")
print(f"创建的批次数量：{batch_count}")
