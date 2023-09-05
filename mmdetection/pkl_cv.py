import os
import pickle
import cv2
import numpy as np
import torch

pre_score = 0.5

alpha = 0.65  # 设置透明度

# 创建保存可视化结果的目录
output_dir = './pkl_cv'
os.makedirs(output_dir, exist_ok=True)

# 读取pkl文件内容
path = './results.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

# 遍历数据进行可视化
for i, item in enumerate(data):
    img_path = item['img_path']
    img = cv2.imread(img_path)

    for j in range(len(item['pred_instances']['bboxes'])):
        # try:
        score = item['pred_instances']['scores'][j]
        bbox = item['pred_instances']['bboxes'][j]
        label = item['pred_instances']['labels'][j]
        keypoints1 = item['pred_instances']['keypoints1'][j]
        keypoints2 = item['pred_instances']['keypoints2'][j]
        
        if score <= pre_score:
            continue

        score = round(score.item() * 100.0, 1)

        # 创建一个与图像相同大小的空白图像作为透明图层
        overlay = img.copy()

        # 绘制边界框
        cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255 - int(label*30), int(label*60), int(label*80)), 2)

        label_name = {0:'fullcar',1:'cyclist',2:'ped',3:'car'}
        
        # 绘制边界框
        cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255 - int(label*30), int(label*60), int(label*80)), 2)

        # 在图像上显示标签信息
        label_text = f'{label_name[int(label)]}:{score}'

        # 计算标签文本的大小
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_width, text_height = text_size[0], text_size[1]

        # 绘制黑色背景矩形
        cv2.rectangle(overlay, (int(bbox[0] + 2), int(bbox[1]) - text_height + 28), (int(bbox[0]) + text_width, int(bbox[1]) + 2), (0, 0, 0), -1)

        # 在黑色背景上绘制白色字样的标签信息
        cv2.putText(overlay, label_text, (int(bbox[0] + 2), int(bbox[1]) + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 在透明图层上添加透明效果
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        
        # 绘制关键点
        # for kp1, kp2 in zip(keypoints1, keypoints2):
        if label == 0 or label == 1:
            x1, y1 = int(keypoints1[0]), int(keypoints1[1])
            x2, y2 = int(keypoints2[0]), int(keypoints2[1])
            # kp1_text = 'rear'
            # kp2_text = 'front'
            cv2.circle(img, (x1, y1), 3, (255, 0, 0), -1)
            cv2.circle(img, (x2, y2), 3, (0, 0, 255), -1)

        # except:
        #     pass
        
    # 保存可视化结果
    output_path = os.path.join(output_dir, f'visualization_{i}.jpg')
    cv2.imwrite(output_path, img)
    print(f'Saved visualization {i} at {output_path}')