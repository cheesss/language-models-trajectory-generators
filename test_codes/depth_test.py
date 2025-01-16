import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()

# 2. 스트리밍 설정 (깊이 데이터 활성화)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 3. 스트리밍 시작
pipeline.start(config)

try:
    while True:
        # 4. 프레임 가져오기
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue

        # 5. 깊이 데이터를 Numpy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())

        # 6. 깊이 데이터를 시각화
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Stream', depth_colormap)

        # ESC 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    # 스트리밍 종료
    pipeline.stop()
    cv2.destroyAllWindows()
