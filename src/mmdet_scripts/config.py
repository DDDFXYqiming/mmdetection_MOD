"""默认配置：路径、类别映射、筛选参数。"""

# 障碍物类型 → 类别 ID
OBS_TYPE_TO_CATEGORY = {
    "ObstacleRawModel_FullCar": 1,
    "ObstacleRawModel_Cyclist": 2,
    "ObstacleRawModel_Ped": 3,
    "ObstacleRawModel_Car": 4,
}

# keypoints 变体的类别（关键点类别）
KEYPOINT_CATEGORIES = {
    "ObstacleRawModel_front": 5,
    "ObstacleRawModel_rear": 6,
}

# 类别名称（COCO 输出）
CATEGORY_NAMES = {
    1: "fullcar",
    2: "cyclist",
    3: "ped",
    4: "car",
}

# 预测可视化标签名
PRED_LABEL_NAMES = {0: "fullcar", 1: "cyclist", 2: "ped", 3: "car"}

# COCO 子目录
COCO_SPLITS = ["train2017", "val2017", "test2017"]

# 默认摄像头参数（距离估算）
DEFAULT_CAMERA = {
    "height": 1.5,      # 摄像头高度（米）
    "focal_length": 1449.0,  # 焦距（像素）
    "obj_height": 1.7,  # 行人/骑行平均高度（米）
    "car_length": 4.5,  # 汽车平均长度（米）
}
