import cv2
import numpy as np
from viewTransformer import  PerspectiveTransformer

# 1. 读取视频
video_path = r"D:\FrisbeeDetection\data\v_d_1.MP4"  
cap = cv2.VideoCapture(video_path)

# 2. 设置场地四个角点（手动调整）
src_points = [(400, 75), (590, 75), (164, 520), (958, 510)]  # 示例值
court_width, court_height = 333, 900  # 目标球场尺寸
dst_points = [(0, 0), (court_width, 0), (0, court_height), (court_width, court_height)]

# 3. 初始化 PerspectiveTransformer 类
perspective_transformer = PerspectiveTransformer(src_points, dst_points, (court_width, court_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 5. 显示原图 & 透视变换后的图像
    scale  =0.5
    resized_frame = cv2.resize(frame,None, fx = scale,fy=scale)
    height, width = resized_frame.shape[:2]
    text = f"{width} x {height}"
    position = (int(width*0.8),int(height*0.05))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # 绿色
    thickness = 2
    cv2.putText(resized_frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    transformed_frame = perspective_transformer.warp_perspective(resized_frame)
    
    for point in src_points:
        cv2.circle(resized_frame, point, 5, (0, 0, 255), -1)  # 画点（红色）

    # 连接成矩形（按顺序连接四个点）
    ordered_idx = [0, 1, 3, 2, 0]  # 按正确顺序连接

    for i in range(4):
        cv2.line(resized_frame, src_points[ordered_idx[i]], src_points[(ordered_idx[i+1])%4], (0, 255, 0), 2)  # 画线（绿色）

        
    # for i in range(4):
    #     cv2.line(resized_frame, src_points[i], src_points[(i+1) % 4], (0, 255, 0), 2)  # 画线（绿色）
        
    cv2.imshow("Original Video", resized_frame)
    cv2.imshow("Transformed Video", transformed_frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
