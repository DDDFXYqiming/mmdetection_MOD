import cv2
import os

frames = 10

# 定义图片文件夹路径和生成的视频保存路径
image_folder = '/home/omnisky/disk14/team/feixiaoyue/mmdetection/data/coco/val2017'
video_name = 'demo_car.mp4'

# 获取图片文件夹中所有以 ".jpg" 结尾的图片文件
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images.sort()  # 对图片文件进行排序
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape  # 获取第一张图片的高度、宽度和通道数

# 使用 "mp4v" 编码器创建一个 VideoWriter 对象，以及设置帧率和视频尺寸
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, frames, (width, height))

# 遍历图片列表，将每张图片写入视频
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# 关闭所有打开的窗口并释放视频写入对象
cv2.destroyAllWindows()
video.release()
