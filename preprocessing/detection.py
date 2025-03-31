import cv2
import numpy as np

# 1. 读取视频
video_path = "input_video.mp4"  # 可替换为摄像头: cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)

# 2. 设置场地四个角点（手动调整）
src_points = [(100, 300), (500, 300), (50, 600), (550, 600)]  # 示例值
court_width, court_height = 600, 800  # 目标球场尺寸
dst_points = [(0, 0), (court_width, 0), (0, court_height), (court_width, court_height)]

# 3. 初始化 PerspectiveTransformer 类
perspective_transformer = PerspectiveTransformer(src_points, dst_points, (court_width, court_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. 透视变换
    transformed_frame = perspective_transformer.warp_perspective(frame)

    # 5. 显示原图 & 透视变换后的图像
    cv2.imshow("Original Video", frame)
    cv2.imshow("Transformed View", transformed_frame)

    # 按 'q' 退出
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
