from io import StringIO
from contextlib import redirect_stdout


messages= {'content': 'The can has been detected with the following details:\n- **Position**: \\([-0.153, -0.057, 0.686]\\)\n- **Dimensions**: \n  - Width: \\(0.106\\) m\n  - Length: \\(0.09\\) m\n  - Height: \\(0.067\\) m\n- **Orientation**: \n  - Along shorter side (length): \\(-0.232\\) radians\n  - Along longer side (width): \\(1.339\\) radians\n\nSince the can is relatively small, I will approach it from the midpoint of its width to ensure a stable grasp. The gripper will need to rotate to align with the narrow side of the can before closing.\n\n### Step-by-Step Plan for the Trajectory:\n1. **Rotate the End-Effector**: Rotate the end-effector by \\(-0.232\\) radians to align with the narrow side of the can.\n2. **Move to Above the Can**: Move the end-effector to a position directly above the can, at a height of \\(0.686 + 0.1 = 0.786\\) m to ensure clearance.\n3. **Lower the Gripper**: Lower the end-effector to the height of the can, which is \\(0.686\\) m.\n4. **Close the Gripper**: Close the gripper to grasp the can.\n5. **Lift the Can**: Raise the end-effector to a height of \\(0.786\\) m to clear the can from the tabletop.\n6. **Move to a Safe Position**: Move the end-effector to a safe position away from the can.\n\nNow, I will generate the code for the first trajectory step, which is to rotate the end-effector.\n\n```python\nimport numpy as np\n\ndef rotate_end_effector(rotation: float) -> list:\n    # Generate trajectory points for rotation\n    points = []\n    for i in range(100):\n        angle = np.linspace(0, rotation, 100)[i]\n        points.append([0.0, 0.6, 0.55, angle])\n    return points\n\ntrajectory_1 = rotate_end_effector(-0.232)\nexecute_trajectory(trajectory_1)\n```'}

    # llm이 전달해준 메세지를 자른다. 
code_block = messages["content"].split("```python")
#   {"role": "assistant", "content": "```python\nprint('Hello, World!')\n```"} 꼴의 데이터에서 'Hello, World!'를 가져온다,
#   코드가 리턴되므로 코드 블럭이라는 변수에 저장해준다.
print("code_block = ", code_block)
block_number = 0
# save_code_block_to_file(code_block)
for block in code_block:
    if len(block.split("```")) > 1:
        print("block.split: ", block.split("```"))
        # print(1111)
        # 생성된 코드문을 받은 후, ''' -------''' 기준으로 쪼개 실행한다.
        code = block.split("```")[0]
        print("code = ", code)
        # save_code_block_to_file(code)
        block_number += 1
        try:
            # f = StringIO()
            # with redirect_stdout(f):
            print(code)
        except Exception:
            # 에러메세지를 다시 전달한다. 
            error = True