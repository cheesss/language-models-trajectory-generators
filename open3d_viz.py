import open3d as o3d
import numpy as np

def draw_boxes_open3d(bounding_boxes, pcd=None, coordinate_frames=None):
    """
    bounding_boxes: 리스트 또는 numpy 배열
        각 요소가 8개의 꼭짓점(8x3 배열)인 bounding box.
    pcd: open3d.geometry.PointCloud 객체 또는 리스트 (선택 사항)
    coordinate_frames: 단일 객체 또는 객체 리스트 (좌표축)
    """
    geometries = []
    
    # pcd 처리 (단일 객체 또는 리스트)
    if pcd is not None:
        if isinstance(pcd, list):
            geometries.extend(pcd)
        else:
            geometries.append(pcd)
    
    # bounding box의 선 연결 순서 (top: 0~3, bottom: 4~7)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    for box in bounding_boxes:
        box = np.array(box)
        if box.shape != (8, 3):
            print("Error: bounding box does not have 8 vertices. Got shape:", box.shape)
            continue
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(box),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])
        geometries.append(line_set)
    
    # coordinate_frames 처리 (단일 객체 또는 리스트)
    if coordinate_frames is not None:
        if isinstance(coordinate_frames, list):
            geometries.extend(coordinate_frames)
        else:
            geometries.append(coordinate_frames)
    
    o3d.visualization.draw_geometries(geometries)
