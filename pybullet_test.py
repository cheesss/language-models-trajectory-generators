import pybullet as p
import numpy as np
import pybullet_data
import time
import math
from config import control_dt, base_start_position_franka, joint_start_positions_franka, ee_index_franka, ee_start_position, global_scaling

def trajectory_to_object():
    start_position = np.array([0.0, 0.6, 0.55])  # 현재 위치
    approach_position = np.array([-0.192, 0.474, 0.55])  # 물체 위
    pre_grasp_position = np.array([-0.192, 0.474, 0.12])  # 물체 바로 위

    # 시간 정의
    total_time = 2.0  # 총 이동 시간 (초)
    time_steps = 100
    t = np.linspace(0, total_time, time_steps)

    # 궤적 생성 (3차 다항식)
    def polynomial_trajectory(start, end, t):
        a0 = start
        a1 = 0
        a2 = 3 * (end - start) / (total_time**2)
        a3 = -2 * (end - start) / (total_time**3)
        return a0 + a1 * t + a2 * t**2 + a3 * t**3

    trajectory = []
    for ti in t:
        if ti <= total_time / 2:
            position = polynomial_trajectory(start_position, approach_position, ti)
        else:
            position = polynomial_trajectory(approach_position, pre_grasp_position, ti - total_time / 2)
        trajectory.append(position.tolist())

    return trajectory

class Environment:
    def __init__(self, mode="default"):
        self.mode = mode
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

    def load(self):
        p.resetDebugVisualizerCamera(1.5, 0, -30, [0.5, 0.5, 0.0])
        plane_id = p.loadURDF("plane.urdf")

        # 물체 초기 위치와 오리엔테이션 설정
        self.object_position = [-0.192, 0.474, 0.1]
        self.object_orientation_q = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.object_model = p.loadURDF(
            "ycb_assets/003_cracker_box.urdf", 
            self.object_position, 
            self.object_orientation_q, 
            useFixedBase=True, 
            globalScaling=global_scaling
        )

    def update(self):
        p.stepSimulation()
        time.sleep(control_dt)

class Robot:
    def __init__(self):
        self.base_start_position = base_start_position_franka
        self.base_start_orientation_q = p.getQuaternionFromEuler([0, 0, 0])
        self.joint_start_positions = joint_start_positions_franka
        self.id = p.loadURDF("franka_robot/panda.urdf", self.base_start_position, self.base_start_orientation_q, useFixedBase=True)
        self.ee_index = ee_index_franka

        # 로봇 초기 관절 상태 설정
        i = 0
        for j in range(p.getNumJoints(self.id)):
            joint_type = p.getJointInfo(self.id, j)[2]
            if joint_type in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
                p.resetJointState(self.id, j, self.joint_start_positions[i])
                i += 1

    def execute_trajectory(self, trajectory):
        for point in trajectory:
            pos = point
            orn = p.getQuaternionFromEuler([0, 0, 0])
            joint_positions = p.calculateInverseKinematics(self.id, self.ee_index, pos)

            for j, pos in enumerate(joint_positions):
                p.setJointMotorControl2(self.id, j, p.POSITION_CONTROL, pos)

            p.stepSimulation()
            time.sleep(control_dt)

# 시뮬레이션 실행
env = Environment()
env.load()
robot = Robot()

# 궤적 생성 및 실행
trajectory = trajectory_to_object()
for i in range(len(trajectory) - 1):
    p.addUserDebugLine(trajectory[i], trajectory[i + 1], lineColorRGB=[1, 0, 0], lineWidth=2)

robot.execute_trajectory(trajectory)  # 물체로 부드럽게 이동

print("Press 'ESC' to exit...")

# 종료 방지
while True:
    keys = p.getKeyboardEvents()
    if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
        print("Simulation terminated.")
        break
    p.stepSimulation()
    time.sleep(control_dt)

p.disconnect()
