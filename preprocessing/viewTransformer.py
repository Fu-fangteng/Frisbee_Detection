import cv2
import numpy as np

class PerspectiveTransformer:
    def __init__(self, src_points, dst_points, court_size):
        """
        初始化透视变换类
        :param src_points: 图像中的四个角点 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        :param dst_points: 目标球场坐标 [(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)]
        :param court_size: 球场尺寸 (width, height)
        """
        self.src_points = np.array(src_points, dtype=np.float32)
        self.dst_points = np.array(dst_points, dtype=np.float32)
        self.court_size = court_size
        
        # 计算透视变换矩阵
        self.transformation_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inverse_transformation_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
    
    def warp_perspective(self, frame):
        """
        将输入图像进行透视变换，转换到标准球场视角
        :param frame: 输入视频帧
        :return: 透视变换后的图像
        """
        return cv2.warpPerspective(frame, self.transformation_matrix, self.court_size)
    
    def transform_point(self, point):
        """
        将图像中的点坐标转换到球场坐标系
        :param point: (x, y) 形式的图像坐标
        :return: (X, Y) 球场坐标系中的坐标 或 None（如果不在场地范围内）
        """
        # 确保点是 NumPy 数组格式
        px = np.array([[point]], dtype=np.float32)
        
        # 透视变换
        transformed = cv2.perspectiveTransform(px, self.transformation_matrix)
        transformed_point = tuple(transformed[0][0])

        # 检查是否在球场范围内（四边形检测）
        in_court = cv2.pointPolygonTest(self.src_points, point, measureDist=False) >= 0
        return transformed_point if in_court else None
    
    def inverse_transform_point(self, point):
        """
        将球场坐标转换回视频图像坐标
        :param point: (X, Y) 球场坐标
        :return: (x, y) 视频图像坐标
        """
        px = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(px, self.inverse_transformation_matrix)
        return tuple(transformed[0][0])
