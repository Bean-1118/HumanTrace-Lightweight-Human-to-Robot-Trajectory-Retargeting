import os
import numpy as np
import pandas as pd
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ============================================================
# Configuration
# ============================================================

HUMAN_WAYPOINT_FILE = "data/processed/demo01_waypoints.csv"

ROBOT_SCALE = 0.10

KP = 8.0

MAX_ACTION = 0.30

POSITION_TOLERANCE = 0.008

MAX_STEPS_PER_WAYPOINT = 40

NUM_WAYPOINTS = 20


# ============================================================
# Load human trajectory only to recover the same final target
# ============================================================

human_waypoints = pd.read_csv(HUMAN_WAYPOINT_FILE)

final_dx = human_waypoints["dx_norm"].iloc[-1]
final_dy = human_waypoints["dy_norm"].iloc[-1]


# ============================================================
# Create LIBERO environment
# ============================================================

benchmark_dict = benchmark.get_benchmark_dict()

suite = benchmark_dict["libero_spatial"]()

task_id = 2
task = suite.get_task(task_id)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file
)

print("Task:")
print(task.language)

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=256,
    camera_widths=256,
)

env.seed(0)

obs = env.reset()

start_pos = obs["robot0_eef_pos"].copy()

print()
print("Robot start EEF:")
print(start_pos)


# ============================================================
# Same endpoint as human demonstration
# ============================================================

end_pos = start_pos.copy()

end_pos[0] += ROBOT_SCALE * final_dx
end_pos[1] += ROBOT_SCALE * final_dy

end_pos[2] = start_pos[2]

print()
print("Straight-line start:")
print(start_pos)

print("Straight-line end:")
print(end_pos)


# ============================================================
# Generate straight-line trajectory
# ============================================================

alphas = np.linspace(
    0.0,
    1.0,
    NUM_WAYPOINTS
)

robot_targets = np.array([
    (1.0 - a) * start_pos + a * end_pos
    for a in alphas
])


# ============================================================
# Output folder
# ============================================================

os.makedirs(
    "results/straight_baseline_frames",
    exist_ok=True
)


# ============================================================
# Follow straight-line trajectory
# ============================================================

actual_positions = []

for waypoint_id, target in enumerate(robot_targets):

    print()
    print(
        f"Waypoint {waypoint_id + 1}/{NUM_WAYPOINTS}"
    )

    for step in range(MAX_STEPS_PER_WAYPOINT):

        current = obs["robot0_eef_pos"].copy()

        error = target - current

        distance_xy = np.linalg.norm(
            error[:2]
        )

        if distance_xy < POSITION_TOLERANCE:
            break

        xyz_command = KP * error

        xyz_command = np.clip(
            xyz_command,
            -MAX_ACTION,
            MAX_ACTION
        )

        action = np.array([
            xyz_command[0],
            xyz_command[1],
            xyz_command[2],

            0.0,
            0.0,
            0.0,

            -1.0
        ])

        obs, reward, done, info = env.step(action)

    final_pos = obs["robot0_eef_pos"].copy()

    actual_positions.append(final_pos)

    print(
        "Target :",
        np.round(target, 4)
    )

    print(
        "Actual :",
        np.round(final_pos, 4)
    )

    print(
        "Error  :",
        round(
            np.linalg.norm(
                target - final_pos
            ),
            4
        ),
        "m"
    )

    if "agentview_image" in obs:

        image = np.flipud(
            obs["agentview_image"]
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            "results/straight_baseline_frames/"
            f"waypoint_{waypoint_id:02d}.png",
            image
        )


# ============================================================
# Save results
# ============================================================

actual_positions = np.array(actual_positions)

results = pd.DataFrame({

    "waypoint":
        np.arange(NUM_WAYPOINTS),

    "target_x":
        robot_targets[:, 0],

    "target_y":
        robot_targets[:, 1],

    "target_z":
        robot_targets[:, 2],

    "actual_x":
        actual_positions[:, 0],

    "actual_y":
        actual_positions[:, 1],

    "actual_z":
        actual_positions[:, 2],
})

results.to_csv(
    "results/robot_straight_trajectory.csv",
    index=False
)


final_error = np.linalg.norm(
    robot_targets[-1] -
    actual_positions[-1]
)

print()
print("==============================")
print("Straight baseline complete")
print("==============================")

print(
    "Final target:",
    np.round(
        robot_targets[-1],
        4
    )
)

print(
    "Final actual:",
    np.round(
        actual_positions[-1],
        4
    )
)

print(
    "Final error:",
    round(final_error, 4),
    "m"
)

env.close()
