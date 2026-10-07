import os
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
suite = benchmark_dict["libero_spatial"]()

task_id = 2
task = suite.get_task(task_id)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file
)

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=256,
    camera_widths=256,
)

env.seed(0)
obs = env.reset()

start_pos = obs["robot0_eef_pos"].copy()

print("Start EEF position:", start_pos)

# 7D action:
# [dx, dy, dz, dRx, dRy, dRz, gripper]
#
# We start with a small +X command.
action = np.array([
    0.2,   # +X
    0.0,   # Y
    0.0,   # Z
    0.0,   # rotation x
    0.0,   # rotation y
    0.0,   # rotation z
   -1.0,   # gripper open
])

for i in range(20):
    obs, reward, done, info = env.step(action)

end_pos = obs["robot0_eef_pos"].copy()

print("End EEF position:  ", end_pos)
print("Delta:             ", end_pos - start_pos)

env.close()

