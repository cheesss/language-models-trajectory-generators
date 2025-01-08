import numpy as np

data = [
    {
        'scores': np.array([0.32972702, 0.3048461], dtype=np.float32),
        'labels': ['can', 'can'],
        'boxes': np.array([[144.782394, 156.510162, 170.160934, 196.057495],
                           [82.925148, -0.0616188, 172.765533, 119.929886]], dtype=np.float32),
        'masks': np.array([[[0., 0., 0., ..., 0., 0., 0.],
                            [0., 0., 0., ..., 0., 0., 0.],
                            [0., 0., 0., ..., 0., 0., 0.],
                            [0., 0., 0., ..., 0., 0., 0.]]]),
        'mask_scores': np.array([0.98607427, 0.9793214], dtype=np.float32)
    }
]

# 주어진 리스트에서 네 개의 요소로 분리
result_dict = data[0]  # 리스트의 첫 번째 딕셔너리 추출
print("result_dict=",result_dict)
scores = result_dict['scores']
labels = result_dict['labels']
boxes = result_dict['boxes']
masks = result_dict['masks']

# 출력 확인
print("Scores:", scores)
print("Labels:", labels)
print("Boxes:", boxes)
print("Masks:", masks.shape)  # 마스크의 차원 확인