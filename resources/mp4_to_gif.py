from moviepy.editor import VideoFileClip

def convert_to_gif(input_path, output_path, fps=10):
    video_clip = VideoFileClip(input_path)
    gif_clip = video_clip.subclip(0, video_clip.duration)
    gif_clip = gif_clip.set_duration(video_clip.duration)
    gif_clip = gif_clip.set_fps(fps)
    gif_clip.write_gif(output_path)

# 输入MP4视频文件路径和输出GIF文件路径
input_video_path = 'xingche2.mp4'
output_gif_path = 'xingche2.gif'

# 调用转换函数
convert_to_gif(input_video_path, output_gif_path)
