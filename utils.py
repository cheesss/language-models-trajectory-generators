# utils.py
import pybullet as p
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import math
import config
from PIL import Image
from torchvision.utils import save_image
from shapely.geometry import MultiPoint, Polygon, polygon
import multiprocessing
import logging
import pyrealsense2 as rs
import open3d as o3d
import copy

depth_scale = 0.0010000000474974513

base2cam = np.array([
    [ 9.97956043e-01,  1.24564979e-03,  6.38919201e-02,  5.63566259e-02],
    [-3.78161693e-02, -7.94446176e-01,  6.06156097e-01, -1.47778554e+00],
    [ 5.15137493e-02, -6.07333292e-01, -7.92775253e-01,  4.75763504e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
])

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.INFO)

def get_segmentation_mask(model_predictions, segmentation_threshold):
    masks = []
    for model_prediction in model_predictions:
        model_prediction_np = model_prediction.detach().cpu().numpy()
        seg_thr = np.max(model_prediction_np) - segmentation_threshold * (np.max(model_prediction_np) - np.min(model_prediction_np))
        model_prediction[model_prediction < seg_thr] = False
        model_prediction[model_prediction >= seg_thr] = True
        masks.append(model_prediction)
    return masks


def get_max_contour(image, image_width, image_height):
    ret, thresh = cv.threshold(image, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    contour_index = None
    max_length = 0
    for c, contour in enumerate(contours):
        contour_points = [
            (c, r)
            for r in range(image_height)
            for c in range(image_width)
            if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1
        ]
        if len(contour_points) > max_length:
            contour_index = c
            max_length = len(contour_points)
    if contour_index is None:
        return None
    return contours[contour_index]


def get_intrinsics_extrinsics(pipeline, image_height, camera_position, camera_orientation_q):
    profile = pipeline.get_active_profile()
    stream = profile.get_stream(rs.stream.color)
    intrinsics = stream.as_video_stream_profile().get_intrinsics()

    K = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

    R = np.array(p.getMatrixFromQuaternion(camera_orientation_q)).reshape(3, 3)
    Rt = np.hstack((R, np.array(camera_position).reshape(3, 1)))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1])))

    return K, Rt


def save_xmem_image(masks):
    xmem_array = np.array(Image.open(config.xmem_input_path).convert("L"))
    xmem_array = np.unique(xmem_array, return_inverse=True)[1].reshape(xmem_array.shape)
    xmem_array = xmem_array / np.max(xmem_array)
    save_image(torch.Tensor(xmem_array), config.xmem_input_path)


def outier_removed_point_cloud(points):
    """
    - points: (N, 3) ndarray
    - 반환: inlier_cloud (open3d.geometry.PointCloud)
      시각화는 여기서 하지 않고, 단순히 PointCloud 객체만 반환.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 다운샘플링 & 이상치 제거
    voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=0.005)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    inlier_cloud = voxel_down_pcd.select_by_index(ind)

    # 카메라 -> base 좌표계로 변환
    inlier_cloud = inlier_cloud.transform(base2cam)

    # ----- [시각화 코드 제거] -----
    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    # c_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # c_frame.transform(base2cam)
    # o3d.visualization.draw_geometries([inlier_cloud, frame, c_frame])
    # --------------------------------

    return inlier_cloud


def get_bounding_cube_from_point_cloud(
    image, masks, depth_array, camera_position, camera_orientation_q,
    depth_image, depth_intrinsics, segmentation_count
):
    """
    반환값:
      bounding_cubes: shape (N, 8, 3)  -> N개의 bounding box (각각 8개 꼭짓점)
      bounding_cubes_orientations: shape (N, 2) -> 각 박스의 orientation
      contour_pixel_points: (디버깅용)
      pcds: 각 마스크별 inlier_cloud (원한다면 추가 반환)
    """
    image_width, image_height = image.size
    print("image_width, image_height: ", image_width, image_height)

    bounding_cubes = []
    bounding_cubes_orientations = []
    pcds = []  # 각 마스크별 PCD를 저장할 리스트

    depth_in_meters = depth_image.astype(np.float32)

    for i, mask in enumerate(masks):
        save_image(mask, config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i))
        mask_np = cv.imread(config.bounding_cube_mask_image_path.format(object=segmentation_count, mask=i), cv.IMREAD_GRAYSCALE)
        plt.imshow(mask_np, cmap='gray')
        plt.title("Mask Image")
        plt.show()

        contour = get_max_contour(mask_np, image_width, image_height)
        if contour is not None:
            print("depth_array.size: ", depth_array.shape)
            print("depth_image.shape: ", depth_image.shape)

            contour_pixel_points = [
                (c, r, depth_array[r][c])
                for r in range(image_height)
                for c in range(image_width)
                if cv.pointPolygonTest(contour, (r, c), measureDist=False) == 1
            ]

            # depth_value = depth_image * depth_scale
            # deproject
            world_points_list = []
            for pixel_point in contour_pixel_points:
                c, r, depth_val = pixel_point
                if depth_val > 0:
                    depth_value = depth_image * depth_scale
                    world_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [r, c], depth_value[c][r])
                    world_points_list.append(world_point)

            if len(world_points_list) == 0:
                continue
            # print("depth_intrinsic: ", depth_intrinsics)

            world_points_list = np.array(world_points_list)

            # outlier 제거 -> PCD 얻기
            inlier_cloud = outier_removed_point_cloud(world_points_list)
            pcds.append(inlier_cloud)  # 리스트에 저장

            # 다시 numpy로 변환
            contour_world_points = np.asarray(inlier_cloud.points)

            max_z_coordinate = np.max(contour_world_points[:, 2])
            min_z_coordinate = np.min(contour_world_points[:, 2])

            depth_offset = 0.07
            top_surface_world_points = [
                wp for wp in contour_world_points if wp[2] > max_z_coordinate - depth_offset
            ]

            rect = MultiPoint([wp[:2] for wp in top_surface_world_points]).minimum_rotated_rectangle
            print(isinstance(rect, Polygon))
            if isinstance(rect, Polygon):
                rect = polygon.orient(rect, sign=-1)
                box = rect.exterior.coords
                box = np.array(box[:-1])
                box_min_x = np.argmin(box[:, 0])
                box = np.roll(box, -box_min_x, axis=0)

                box_top = [list(pt) + [max_z_coordinate] for pt in box]
                box_btm = [list(pt) + [min_z_coordinate] for pt in box]
                bounding_cubes.append(box_top + box_btm)

                # Orientation
                bounding_cubes_orientation_width = np.arctan2(
                    box[1][1] - box[0][1],
                    box[1][0] - box[0][0]
                )
                bounding_cubes_orientation_length = np.arctan2(
                    box[2][1] - box[1][1],
                    box[2][0] - box[1][0]
                )
                bounding_cubes_orientations.append([
                    bounding_cubes_orientation_width,
                    bounding_cubes_orientation_length
                ])

    bounding_cubes = np.array(bounding_cubes)
    print(bounding_cubes)
    # contour_pixel_points는 마지막 contour만 반환하거나,
    # 필요하면 별도 리스트에 저장해서 모두 반환해도 됨
    return bounding_cubes, bounding_cubes_orientations, contour_pixel_points, pcds
