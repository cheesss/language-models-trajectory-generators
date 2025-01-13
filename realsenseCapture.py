import threading
import cv2
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from PIL import Image
import multiprocessing
import logging

class IntelCamera:
    def __init__(self, image):
        self.image = image
        
    def save_RGB_img(image):
        # RGB 이미지를 JPEG 파일로 저장
        cv2.imwrite("captured_image.png", image)
        print("RGB image saved as 'captured_image.png'")

    def save_Depth_img(depth_image):
        
        cv2.imwrite("captured_depth_image.png", depth_image)
        print("Depth image saved as 'captured_depth_image.png'")
    
    def capture_save_image():

        # Logging
        # logger = multiprocessing.log_to_stderr()
        # logger.setLevel(logging.INFO)

        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {depth_scale} meters per unit")
        align = rs.align(rs.stream.color)

        try:
            # pipeline.start(config)

            # 프레임 읽기
            for _ in range(30):
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

            if not color_frame:
                print("No color frame captured!")
                return

            color_image = np.asanyarray(color_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())
            colorizer = rs.colorizer(color_scheme = 2)
            colored_depth_frame = colorizer.colorize(depth_frame)
            colored_depth_image = np.asanyarray(colored_depth_frame.get_data())
            threading.Thread(target=IntelCamera.save_RGB_img, args=(color_image,)).start()
            threading.Thread(target=IntelCamera.save_Depth_img, args=(colored_depth_image,)).start()

            # 굳이 필요없음
            # cv2.imshow('Captured Image', color_image)

            # while True:
            #     key = cv2.waitKey(1) 
            #     if key == 27:  # ESC 키를 누르면 종료
            #         break
            #     if cv2.getWindowProperty('Captured Image', cv2.WND_PROP_VISIBLE) < 1:
            #         break

            # # 창 닫기
            # cv2.destroyAllWindows()

    # 리턴값 없이 촬영한 사진을 저장한 후 불러와 사용한다.
        finally:
            pipeline.stop()