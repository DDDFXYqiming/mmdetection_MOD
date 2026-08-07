"""图片序列 → 视频（原 mp4.py）。"""

import os

import cv2


def images_to_video(image_folder: str, video_name: str = "demo_car.mp4", fps: int = 10) -> int:
    images = sorted(f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png")))
    if not images:
        print("没有找到图片")
        return 0
    first = cv2.imread(os.path.join(image_folder, images[0]))
    if first is None:
        print("无法读取第一张图片")
        return 0
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        if frame is not None:
            video.write(frame)
    video.release()
    print(f"视频生成完成：{video_name}，共 {len(images)} 帧，{fps}fps")
    return len(images)
