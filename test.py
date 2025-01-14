from realsenseCapture import IntelCamera
from utils import get_bounding_cube_from_point_cloud
from utils import get_segmentation_mask
import numpy as np
from PIL import Image
import config
import math
import models
from lang_sam import LangSAM as langsam_model

IntelCamera.capture_save_image()


rgb_image_head = Image.open("/home/vlm/language-models-trajectory-generators/captured_image.png").convert("RGB")

model_predictions, boxes, segmentation_texts = models.get_langsam_output(rgb_image_head, langsam_model, segmentation_texts=["can"], segmentation_count=0)


masks = get_segmentation_mask(model_predictions, config.segmentation_threshold)
depth_image_head = Image.open(config.depth_image_head_path).convert("L")
depth_array = np.array(depth_image_head) / 255.
head_camera_position = [0.0, 1.2, 0.6]
head_camera_orientation_e = [0.0, 3 / 4.5 * math.pi, -math.pi / 2]



a, b = get_bounding_cube_from_point_cloud(rgb_image_head, masks, depth_array, head_camera_position, head_camera_orientation_e, segmentation_count=0)
print(a, b)