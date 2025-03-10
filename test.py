from realsenseCapture import IntelCamera
from utils import get_bounding_cube_from_point_cloud, get_segmentation_mask, base2cam
import utils
import numpy as np
from PIL import Image
import config
import math
import models
from lang_sam import LangSAM 
import pybullet as p
import pybullet_data
import time
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

# open3d 시각화 함수 임포트
from open3d_viz import draw_boxes_open3d

# 캡처 및 모델 준비
rgba_image, depth_image, depth_intrinsics = IntelCamera.capture_save_image()
langsam_model = LangSAM()
rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")

model_predictions, boxes, segmentation_texts = models.get_langsam_output(
    rgb_image_head, langsam_model, segmentation_texts=["red box"], segmentation_count=0
)

masks = get_segmentation_mask(model_predictions, config.segmentation_threshold)
depth_image_head = Image.open(config.depth_image_head_path).convert("L")
depth_scale = 0.0010000000474974513
depth_array = np.array(depth_image_head) * depth_scale

head_camera_position = [0.0, 1.2, 0.6]
head_camera_orientation_e = [0.0, 5 / 6 * math.pi, -math.pi / 2]
camera_orientation_q = p.getQuaternionFromEuler(config.head_camera_orientation_e)

# RealSense 파이프라인 시작
pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(rs_config)

try:
    # get_bounding_cube_from_point_cloud 함수 호출 (추가로 PCD도 반환)
    bounding_cubes_world_coordinates, bounding_cubes_orientations, contour_pixel_points, pcds = get_bounding_cube_from_point_cloud(
        rgb_image_head, masks, depth_array, head_camera_position, camera_orientation_q, 
        depth_image=depth_image, depth_intrinsics=depth_intrinsics, segmentation_count=0
    )
finally:
    pipeline.stop()

# 반환값 디버깅
bounding_cubes_array = np.array(bounding_cubes_world_coordinates)
print("Debug: Bounding cubes shape:", bounding_cubes_array.shape)
print("Debug: Bounding cubes data:", bounding_cubes_array)

# 크기 및 방향 출력 함수
def check_real_size():
    for i, cube in enumerate(bounding_cubes_world_coordinates):
        cube[4][2] -= config.depth_offset
        object_width = np.around(np.linalg.norm(cube[1] - cube[0]), 3)
        object_length = np.around(np.linalg.norm(cube[2] - cube[1]), 3)
        object_height = np.around(np.linalg.norm(cube[5] - cube[0]), 3)
        print("Position of", segmentation_texts[i], ":", list(np.around(cube[4], 3)))
        print("Dimensions:")
        print("Width:", object_width)
        print("Length:", object_length)
        print("Height:", object_height)
        if object_width < object_length:
            print("Orientation along shorter side (width):", np.around(bounding_cubes_orientations[i][0], 3))
            print("Orientation along longer side (length):", np.around(bounding_cubes_orientations[i][1], 3), "\n")
        else:
            print("Orientation along shorter side (length):", np.around(bounding_cubes_orientations[i][1], 3))
            print("Orientation along longer side (width):", np.around(bounding_cubes_orientations[i][0], 3), "\n")

check_real_size()

# 두 개의 좌표축 생성 (원래처럼)
coordinate_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
coordinate_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
coordinate_frame2.transform(base2cam)

# open3d 시각화를 위한 최종 호출 (두 개의 좌표축을 리스트로 전달)
draw_boxes_open3d(bounding_cubes_world_coordinates, pcds, coordinate_frames=[coordinate_frame1, coordinate_frame2])
