import threading
import cv2
import pyrealsense2 as rs
import numpy as np

def save_image(image):
    # 파일 저장 작업을 메인 스레드에서 처리
    cv2.imwrite("captured_image.jpg", image)
    print("Image saved as 'captured_image.jpg'")

def capture_save_image():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        pipeline.start(config)

        # 프레임 읽기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("No color frame captured!")
            return

        color_image = np.asanyarray(color_frame.get_data())

        threading.Thread(target=save_image, args=(color_image,)).start()

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


    finally:
        pipeline.stop()