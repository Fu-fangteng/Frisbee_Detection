import cv2
import numpy as np
from viewTransformer import  PerspectiveTransformer

# video path
video_path = r"D:\FrisbeeDetection\data\v_d_1.MP4"  
cap = cv2.VideoCapture(video_path)

# Manuly set up the court anchoring points
src_points = [(400, 75), (590, 75), (164, 520), (958, 510)]  
court_width, court_height = 333, 900  # target court info. (Scaled by 0.9 originally [37,100])
dst_points = [(0, 0), (court_width, 0), (0, court_height), (court_width, court_height)]


perspective_transformer = PerspectiveTransformer(src_points, dst_points, (court_width, court_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Show the size of video 
    scale  =0.5
    resized_frame = cv2.resize(frame,None, fx = scale,fy=scale)
    height, width = resized_frame.shape[:2]
    text = f"{width} x {height}"
    position = (int(width*0.8),int(height*0.05))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0) 
    thickness = 2
    cv2.putText(resized_frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    
    #Transform frame
    transformed_frame = perspective_transformer.warp_perspective(resized_frame)
    
    
    # Draw boxes and anchoring points
    for point in src_points:
        cv2.circle(resized_frame, point, 5, (0, 0, 255), -1)  
        
    ordered_idx = [0, 1, 3, 2, 0]  
    for i in range(4):
        cv2.line(resized_frame, src_points[ordered_idx[i]], src_points[(ordered_idx[i+1])%4], (0, 255, 0), 2)  # 画线（绿色）


    # Test point transform function
    player = (500, 300)  
    tran_player = perspective_transformer.transform_point(player)
    if tran_player:
        cv2.circle(resized_frame, (int(player[0]), int(player[1])), 5, (255, 0, 0), -1)  # 蓝色原点
        cv2.circle(transformed_frame, (int(tran_player[0]), int(tran_player[1])), 5, (0, 255, 255), -1)  # 黄色转换后点
    
    
    cv2.imshow("Original Video", resized_frame)
    cv2.imshow("Transformed Video", transformed_frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
