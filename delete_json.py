import os

def delete_json_files_in_coco_subdirs(coco_dir):
    subdirs = ["train2017", "test2017", "val2017"]

    # 遍历子目录
    for subdir in subdirs:
        subdir_path = os.path.join(coco_dir, subdir)
        if os.path.exists(subdir_path):
            # 遍历子目录中的文件
            for file in os.listdir(subdir_path):
                if file.endswith(".json"):
                    file_path = os.path.join(subdir_path, file)
                    os.remove(file_path)  # 删除以 ".json" 结尾的文件
                    print(f"已删除：{file_path}")

if __name__ == "__main__":
    coco_dir = "./mmdetection-main/data/coco"
    delete_json_files_in_coco_subdirs(coco_dir)
