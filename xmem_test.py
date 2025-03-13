from models import get_xmem_output
import torch
from XMem.model.network import XMem
import config

device = torch.device("cuda")
xmem_model = XMem(config.xmem_config, "./XMem/saves/XMem.pth", device).eval().to(device)
trajectory_length = 5 # 테스트코드 전 실행한 파이불렛 상의 촬영본이다. 만약 파이불렛 상에서의 시뮬레이션 이후 이 테스트코드를 실행한다면 해당 변수를 수정해야한다.
a = get_xmem_output(model=xmem_model, device=device, trajectory_length=trajectory_length)
print(a)