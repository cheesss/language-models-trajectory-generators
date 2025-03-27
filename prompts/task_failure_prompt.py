# INPUT: [INSERT TASK SUMMARY]
TASK_FAILURE_PROMPT = \
"""The task was not completed successfully, and it needs to be replanned and retried.

Please carefully review the previously generated code and the trajectory that was executed. Based on the original planning instructions and constraints (such as object dimensions, gripper limits, and physical feasibility), do the following:

1. Summarise the trajectory that was executed, highlighting key waypoint poses of the end-effector (including position, orientation, and gripper state). Explain how each step contributed to or failed to achieve the intended task.

2. Summarise the most recent known positions, orientations, and dimensions of each relevant object involved in the task.

3. If there were previous failed attempts, include a brief explanation of what went wrong in those attempts.

4. Suggest what was wrong with the plans for the trajectories, and propose specific corrections that would address these issues (e.g., incorrect gripper orientation, missing grasp step, unreachable pose, etc.).

5. Then, replan and retry the task by continuing with **INITIAL PLANNING 1** from the original prompt. Make sure your reasoning is step-by-step and grounded in physical feasibility, and regenerate the trajectory code accordingly.

Ensure that the new plan satisfies all physical and logical requirements before generating the code.

"""
