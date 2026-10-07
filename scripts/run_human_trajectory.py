import os
import numpy as np
import pandas as pd
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ============================================================
# Configuration
# ============================================================

WAYPOINT_FILE = "data/processed/demo01_waypoints.csv"

# Convert normalized human motion into robot workspace motion.
# 0.10 means the largest normalized displacement becomes ~10 cm.
ROBOT_SCALE = 0.10

# Simple proportional controller gain
KP = 8.0

# Maximum OSC command magnitude
MAX_ACTION = 0.30

# Robot position tolerance for each waypoint
POSITION_TOLERANCE = 0.008   # 8 mm

# Maximum simulation steps spent trying to reach one waypoint
MAX_STEPS_PER_WAYPOINT = 40


# ============================================================
# Load human trajectory
# ============================================================

waypoints = pd.read_csv(WAYPOINT_FILE)

human_dx = waypoints["dx_norm"].to_numpy()
human_dy = waypoints["dy_norm"].to_numpy()

print("Loaded human trajectory:")
print("Number of waypoints:", len(waypoints))


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

print()
print("Task:")
print(task.language)

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=256,
    camera_widths=256,
)

env.seed(0)

obs = env.reset()


# ============================================================
# Initial robot position
# ============================================================

start_pos = obs["robot0_eef_pos"].copy()

print()
print("Robot start EEF:")
print(start_pos)


# ============================================================
# Convert human trajectory to robot trajectory
#
# Human normalized:
#     dx_norm
#     dy_norm
#
# Robot:
#     X <- human X
#     Y <- human Y
#     Z fixed
#
# We deliberately do NOT transfer human depth yet.
# ============================================================

robot_targets = []

for dx, dy in zip(human_dx, human_dy):

    target = start_pos.copy()

    target[0] += ROBOT_SCALE * dx
    target[1] += ROBOT_SCALE * dy

    # Keep height constant
    target[2] = start_pos[2]

    robot_targets.append(target)

robot_targets = np.array(robot_targets)


print()
print("Robot trajectory:")
print("Start target:", robot_targets[0])
print("End target:  ", robot_targets[-1])

print()
print(
    "Maximum XY movement:",
    np.max(
        np.linalg.norm(
            robot_targets[:, :2] - start_pos[:2],
            axis=1
        )
    ),
    "m"
)


# ============================================================
# Save first camera image for debugging
# ============================================================

os.makedirs("results/human_follow_frames", exist_ok=True)

if "agentview_image" in obs:

    image = obs["agentview_image"]

    # LIBERO image orientation may appear vertically flipped.
    image = np.flipud(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        "results/human_follow_frames/start.png",
        image
    )


# ============================================================
# Follow human trajectory
# ============================================================

actual_positions = []

for waypoint_id, target in enumerate(robot_targets):

    print()
    print(
        f"Waypoint {waypoint_id + 1}/{len(robot_targets)}"
    )

    for step in range(MAX_STEPS_PER_WAYPOINT):

        current = obs["robot0_eef_pos"].copy()

        error = target - current

        distance_xy = np.linalg.norm(
            error[:2]
        )

        # Waypoint reached
        if distance_xy < POSITION_TOLERANCE:
            break

        # Proportional controller
        xyz_command = KP * error

        xyz_command = np.clip(
            xyz_command,
            -MAX_ACTION,
            MAX_ACTION
        )

        action = np.array([
            xyz_command[0],
            xyz_command[1],

            # Keep Z controlled as well
            xyz_command[2],

            # No orientation changes
            0.0,
            0.0,
            0.0,

            # Keep gripper open
            -1.0
        ])

        obs, reward, done, info = env.step(action)

    final_pos = obs["robot0_eef_pos"].copy()

    actual_positions.append(final_pos)

    print("Target :", np.round(target, 4))
    print("Actual :", np.round(final_pos, 4))
    print(
        "Error  :",
        round(
            np.linalg.norm(target - final_pos),
            4
        ),
        "m"
    )

    # Save camera frame
    if "agentview_image" in obs:

        image = np.flipud(
            obs["agentview_image"]
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            f"results/human_follow_frames/"
            f"waypoint_{waypoint_id:02d}.png",
            image
        )


# ============================================================
# Save actual robot trajectory
# ============================================================

actual_positions = np.array(actual_positions)

result = pd.DataFrame({

    "waypoint": np.arange(len(robot_targets)),

    "target_x": robot_targets[:, 0],
    "target_y": robot_targets[:, 1],
    "target_z": robot_targets[:, 2],

    "actual_x": actual_positions[:, 0],
    "actual_y": actual_positions[:, 1],
    "actual_z": actual_positions[:, 2],
})

result.to_csv(
    "results/robot_human_trajectory.csv",
    index=False
)


# ============================================================
# Final summary
# ============================================================

final_error = np.linalg.norm(
    robot_targets[-1] -
    actual_positions[-1]
)

print()
print("==============================")
print("Trajectory complete")
print("==============================")

print(
    "Final target:",
    np.round(robot_targets[-1], 4)
)

print(
    "Final actual:",
    np.round(actual_positions[-1], 4)
)

print(
    "Final error:",
    round(final_error, 4),
    "m"
)

print()
print(
    "Saved trajectory to:"
)

print(
    "results/robot_human_trajectory.csv"
)

print(
    "Saved rendered frames to:"
)

print(
    "results/human_follow_frames/"
)

env.close()
