import cv2
import numpy as np
from viewTransformer import PerspectiveTransformer

# 视频路径
video_path = r"D:\FrisbeeDetection\data\v_d_1.MP4"
cap = cv2.VideoCapture(video_path)

# 初始化参数
src_points = []  # 存储用户点击的四个点
court_width, court_height = 333, 900  # 目标球场尺寸
dst_points = [(0, 0), (court_width, 0), (0, court_height), (court_width, court_height)]
perspective_transformer = None  # 透视变换对象
scale = 0.5  # 缩放比例

# 读取第一帧并暂停
ret, frame = cap.read()
if not ret:
    print("Error: 无法读取视频")
    cap.release()
    cv2.destroyAllWindows()
    exit()

resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)
copy_frame = resized_frame.copy()  # 备份第一帧


# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global src_points, copy_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < 4:
            src_points.append((x, y))
            cv2.circle(copy_frame, (x, y), 5, (0, 0, 255), -1)  # 画红点
            cv2.imshow("Select Court Points", copy_frame)

        if len(src_points) == 4:
            print(f"选定点: {src_points}")
            cv2.destroyWindow("Select Court Points")


# 让用户点击四个点
cv2.imshow("Select Court Points", copy_frame)
cv2.setMouseCallback("Select Court Points", mouse_callback)

# 等待用户点击四个点
while len(src_points) < 4:
    cv2.waitKey(1)

# 生成透视变换对象
perspective_transformer = PerspectiveTransformer(src_points, dst_points, (court_width, court_height))

# 开始播放视频
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # 透视变换
    transformed_frame = perspective_transformer.warp_perspective(resized_frame)

    # 画用户选定的四个点
    for point in src_points:
        cv2.circle(resized_frame, point, 5, (0, 0, 255), -1)  # 画红点

    # 画出场地边界
    ordered_idx = [0, 1, 3, 2, 0]
    for i in range(4):
        cv2.line(resized_frame, src_points[ordered_idx[i]], src_points[ordered_idx[i+1]], (0, 255, 0), 2)

    # 显示视频
    cv2.imshow("Original Video", resized_frame)
    cv2.imshow("Transformed Video", transformed_frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
