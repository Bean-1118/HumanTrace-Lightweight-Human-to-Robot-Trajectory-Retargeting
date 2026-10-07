import os

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

print("Task name:", task.name)
print("Language:", task.language)
print("BDDL full path:", task_bddl_file)
print("BDDL exists:", os.path.exists(task_bddl_file))

env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 256,
    "camera_widths": 256,
}

print("Creating environment...")

env = OffScreenRenderEnv(**env_args)

print("Environment created.")

env.seed(0)

obs = env.reset()

print("Environment reset OK.")
print("Observation keys:")

for key in obs.keys():
    print("  ", key)

env.close()

print("Done.")
