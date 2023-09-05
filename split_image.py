import os
import json
import random
import shutil

def split_dataset(json_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, max_time_diff_minutes=1):
    print('正在尝试分割数据集...')
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

    # 计算分割图片数量
    # print(image_groups)
    num_train = int(total_images*train_ratio)
    num_val = int(total_images*val_ratio)
    real_train = 0
    real_val = 0

    print('训练集数量: {}, 验证集数量: {}, 测试集数量: {}'.format(num_train, num_val, (total_images - num_val - num_train)))

    # 根据组的数量将图片分配到对应的集合（训练集、验证集、测试集）中
    for i, group in enumerate(image_groups):
        print("当前组图片数："+str(len(image_groups[i])))
        # if num_train<0:
        #     print("train分配完成")    
        # if num_val<0:
        #     print("val分配完成")


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
                json_file = image_file[:-4] + ".json"
                src_image_path = os.path.join(json_dir, image_file)
                src_json_path = os.path.join(json_dir, json_file)
                
                dst_image_path = os.path.join(test_dir, image_file)
                dst_json_path = os.path.join(test_dir, json_file)

                shutil.copy(src_image_path, dst_image_path)
                shutil.copy(src_json_path, dst_json_path)
    print('实际分配训练集数量: {}, 验证集数量: {}, 测试集数量: {}'.format(real_train, real_val, (total_images - real_val - real_train)))


        
if __name__ == "__main__":
    # json_dir = "./J2camera"
    # json_dir = "./J2camera_selected_50"
    json_dir = "./J2camera_selected_50"
    output_dir = "./data2"
    split_dataset(json_dir, output_dir)
