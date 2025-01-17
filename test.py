from realsenseCapture import IntelCamera
from utils import get_bounding_cube_from_point_cloud
from utils import get_segmentation_mask
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



rgba_image, depth_image, depth_intrinsics = IntelCamera.capture_save_image()
langsam_model = LangSAM()

rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")

model_predictions, boxes, segmentation_texts = models.get_langsam_output(rgb_image_head,langsam_model, segmentation_texts=["banana."], segmentation_count=0)


masks = get_segmentation_mask(model_predictions, config.segmentation_threshold)
# print(1)
depth_image_head = Image.open(config.depth_image_head_path).convert("L")
# print(2)
depth_scale = 0.0010000000474974513
depth_array = np.array(depth_image_head) * depth_scale



# print("depth_array: ",depth_array)
# print(3)
head_camera_position = [0.0, 1.2, 0.6]
# print(4)
# head_camera_orientation_e = [0.0, 5 / 6 * math.pi, -math.pi / 2]
head_camera_orientation_e = [0.0, 5 / 6 * math.pi, -math.pi / 2]

# print(5)
camera_orientation_q = p.getQuaternionFromEuler(config.head_camera_orientation_e)
# print(6)
# bounding_cubes_world_coordinates, bounding_cubes_orientations = utils.get_bounding_cube_from_point_cloud(rgb_image_head, masks, depth_array, head_camera_position, camera_orientation_q, segmentation_count=0)
# print(7)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    # 필요한 작업 수행
    bounding_cubes_world_coordinates, bounding_cubes_orientations, contour_pixel_points = get_bounding_cube_from_point_cloud(
        rgb_image_head, masks, depth_array, head_camera_position, camera_orientation_q, depth_image=depth_image ,depth_intrinsics = depth_intrinsics , segmentation_count=0
    )



finally:
    # 파이프라인 정리
    pipeline.stop()

# a, b =bounding_cubes_world_coordinates, bounding_cubes_orientations 
# print("bounding_cubes_world_coordinates, bounding_cubes_orientations: ",a, b)





# def visualize_depth_array(depth_array):
#     rows, cols = depth_array.shape
#     x = np.linspace(0, cols - 1, cols)
#     y = np.linspace(0, rows - 1, rows) 
#     x, y = np.meshgrid(x, y)
#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     surface = ax.plot_surface(x, y, -depth_array, cmap='viridis', edgecolor='none')

#     # 색상 바 추가
#     fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

#     # 제목과 라벨
#     ax.set_title('3D Surface Plot')
#     ax.set_xlabel('X axis')
#     ax.set_ylabel('Y axis')
#     ax.set_zlabel('Z axis')

#     plt.show()



def update():
    p.stepSimulation()
    time.sleep(config.control_dt)




import config
def check_real_size():
    for i, bounding_cube_world_coordinates in enumerate(bounding_cubes_world_coordinates):

        bounding_cube_world_coordinates[4][2] -= config.depth_offset

        object_width = np.around(np.linalg.norm(bounding_cube_world_coordinates[1] - bounding_cube_world_coordinates[0]), 3)
        object_length = np.around(np.linalg.norm(bounding_cube_world_coordinates[2] - bounding_cube_world_coordinates[1]), 3)
        object_height = np.around(np.linalg.norm(bounding_cube_world_coordinates[5] - bounding_cube_world_coordinates[0]), 3)

        print("Position of", segmentation_texts[i], ":", list(np.around(bounding_cube_world_coordinates[4], 3)))


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




def draw_box_pybullet():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(config.camera_distance, config.camera_yaw, config.camera_pitch, config.camera_target_position)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")

    while True:
        for bounding_cube_world_coordinates in bounding_cubes_world_coordinates:
            p.addUserDebugLine(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[1], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[2], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[3], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[0], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[5], bounding_cube_world_coordinates[6], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[6], bounding_cube_world_coordinates[7], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[7], bounding_cube_world_coordinates[8], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[8], bounding_cube_world_coordinates[5], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[0], bounding_cube_world_coordinates[5], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[1], bounding_cube_world_coordinates[6], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[2], bounding_cube_world_coordinates[7], [0, 1, 0], lifeTime=0)
            p.addUserDebugLine(bounding_cube_world_coordinates[3], bounding_cube_world_coordinates[8], [0, 1, 0], lifeTime=0)
            p.addUserDebugPoints(bounding_cube_world_coordinates, [[0, 1, 0]] * len(bounding_cube_world_coordinates), pointSize=5, lifeTime=0)
        update()

# api.py 내부에 물체 높이 계산 코드 존재    def update(self):


check_real_size()

import config


# visualize_depth_array(depth_array)


# print("contour_pixel_points: ",contour_pixel_points)
# draw_box_pybullet()

