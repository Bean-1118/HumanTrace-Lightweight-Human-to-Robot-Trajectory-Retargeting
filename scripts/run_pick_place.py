
import os
import numpy as np
import pandas as pd
import cv2

from libero.libero import benchmark, get_libero_path

from libero.libero.envs import OffScreenRenderEnv


# ============================================================
# PATHS
# ============================================================

GRASP_PRIMITIVE_PATH = (
    "data/robot_primitives/"
    "panda_grasp_demo0.npz"
)

WAYPOINT_PATH = (
    "data/processed/demo01_waypoints.csv"
)

OUTPUT_DIR = (
    "results/final_pick_place"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

# Official grasp + lift only
GRASP_END_FRAME = 55

GRIPPER_CLOSE = +1.0

KP = 6.0
MAX_ACTION = 0.18

POSITION_TOLERANCE = 0.008
MAX_STEPS_PER_WAYPOINT = 80

# Keep bowl safely above table / plate during transport
TRANSPORT_LIFT_EXTRA = 0.00


# ============================================================
# IMAGE HELPER
# ============================================================

def save_image(obs, name):

    image = np.flipud(
        obs["agentview_image"]
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            name
        ),
        image
    )


# ============================================================
# MOVE EEF TO ONE XYZ TARGET
# ============================================================

def move_to(
    env,
    obs,
    target,
    name
):

    reached = False

    for step in range(
        MAX_STEPS_PER_WAYPOINT
    ):

        current = obs[
            "robot0_eef_pos"
        ].copy()

        error = (
            target - current
        )

        distance = (
            np.linalg.norm(error)
        )

        if distance < POSITION_TOLERANCE:

            reached = True
            break

        command = np.clip(
            KP * error,
            -MAX_ACTION,
            MAX_ACTION
        )

        action = np.array([
            command[0],
            command[1],
            command[2],

            # Hold orientation approximately constant
            0.0,
            0.0,
            0.0,

            GRIPPER_CLOSE,
        ])

        obs, reward, done, info = (
            env.step(action)
        )

    print(
        name,
        "| target:",
        np.round(target, 4),
        "| actual:",
        np.round(
            obs["robot0_eef_pos"],
            4
        ),
        "| error:",
        round(
            np.linalg.norm(
                target -
                obs["robot0_eef_pos"]
            ),
            4
        ),
        "| reached:",
        reached
    )

    return obs


# ============================================================
# LOAD OFFICIAL DEMO
# ============================================================
primitive = np.load(
    GRASP_PRIMITIVE_PATH
)

initial_state = primitive[
    "initial_state"
]

actions = primitive[
    "grasp_actions"
]
# ============================================================
# LOAD HUMAN WAYPOINTS
# ============================================================

human_df = pd.read_csv(
    WAYPOINT_PATH
)

human_xy = human_df[
    ["dx_norm", "dy_norm"]
].to_numpy(
    dtype=float
)

# Force first waypoint to exactly zero
human_xy = (
    human_xy
    - human_xy[0]
)


print()
print("=" * 70)
print("HUMAN TRAJECTORY")
print("=" * 70)

print(
    "Waypoints:",
    len(human_xy)
)

print(
    "Human endpoint:",
    np.round(
        human_xy[-1],
        4
    )
)


# ============================================================
# CREATE LIBERO ENV
# ============================================================

benchmark_dict = (
    benchmark.get_benchmark_dict()
)

suite = benchmark_dict[
    "libero_spatial"
]()

task = suite.get_task(2)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file
)


env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,

    camera_heights=256,
    camera_widths=256,

    horizon=3000,
    ignore_done=True,
)

env.seed(0)

obs = env.reset()


# ============================================================
# RESTORE OFFICIAL INITIAL STATE
# ============================================================

obs = env.set_init_state(
    initial_state
)


print()
print("=" * 70)
print("INITIAL STATE")
print("=" * 70)

print(
    "EEF:",
    np.round(
        obs["robot0_eef_pos"],
        4
    )
)

print(
    "Bowl:",
    np.round(
        obs["akita_black_bowl_1_pos"],
        4
    )
)

print(
    "Plate:",
    np.round(
        obs["plate_1_pos"],
        4
    )
)

save_image(
    obs,
    "00_start.png"
)


# ============================================================
# STAGE 1
# OFFICIAL GRASP PRIMITIVE
# ============================================================

print()
print("=" * 70)
print("STAGE 1: OFFICIAL GRASP + LIFT")
print("=" * 70)

for t in range(
    GRASP_END_FRAME + 1
):

    obs, reward, done, info = (
        env.step(
            actions[t]
        )
    )

    # Save intermediate grasp / lift frames
    # so the GIF shows continuous motion instead of a jump.
    if t % 3 == 0 or t == GRASP_END_FRAME:

        save_image(
            obs,
            f"grasp_{t:03d}.png"
        )

save_image(
    obs,
    "01_after_grasp.png"
)


eef_start = obs[
    "robot0_eef_pos"
].copy()

bowl_start = obs[
    "akita_black_bowl_1_pos"
].copy()

plate_pos = obs[
    "plate_1_pos"
].copy()


print(
    "EEF after grasp:",
    np.round(
        eef_start,
        4
    )
)

print(
    "Bowl after grasp:",
    np.round(
        bowl_start,
        4
    )
)

print(
    "Plate:",
    np.round(
        plate_pos,
        4
    )
)


# ============================================================
# STAGE 2
# COMPUTE HUMAN → ROBOT SIMILARITY TRANSFORM
# ============================================================

print()
print("=" * 70)
print("STAGE 2: HUMAN → ROBOT RETARGETING")
print("=" * 70)


# Human trajectory endpoint vector
human_end = (
    human_xy[-1]
)

human_length = (
    np.linalg.norm(
        human_end
    )
)


# Robot desired planar displacement:
# current bowl XY → plate XY
robot_vector = (
    plate_pos[:2]
    - bowl_start[:2]
)

robot_length = (
    np.linalg.norm(
        robot_vector
    )
)


# ------------------------------------------------------------
# Compute angles
# ------------------------------------------------------------

human_angle = np.arctan2(
    human_end[1],
    human_end[0]
)

robot_angle = np.arctan2(
    robot_vector[1],
    robot_vector[0]
)

rotation_angle = (
    robot_angle
    - human_angle
)


# ------------------------------------------------------------
# 2D rotation matrix
# ------------------------------------------------------------

c = np.cos(
    rotation_angle
)

s = np.sin(
    rotation_angle
)

R = np.array([
    [c, -s],
    [s,  c],
])


# ------------------------------------------------------------
# Scale endpoint magnitude
# ------------------------------------------------------------

scale = (
    robot_length
    / human_length
)


# ------------------------------------------------------------
# Transform human trajectory
# ------------------------------------------------------------

transformed_xy = []

for point in human_xy:

    transformed = (
        scale
        * (R @ point)
    )

    transformed_xy.append(
        transformed
    )


transformed_xy = np.array(
    transformed_xy
)


print(
    "Human endpoint vector:",
    np.round(
        human_end,
        4
    )
)

print(
    "Robot bowl→plate vector:",
    np.round(
        robot_vector,
        4
    )
)

print(
    "Rotation:",
    round(
        np.degrees(
            rotation_angle
        ),
        2
    ),
    "degrees"
)

print(
    "Scale:",
    round(
        scale,
        4
    )
)

print(
    "Transformed endpoint:",
    np.round(
        transformed_xy[-1],
        4
    )
)


# ============================================================
# STAGE 3
# FOLLOW HUMAN-DERIVED TRANSPORT PATH
# ============================================================

print()
print("=" * 70)
print("STAGE 3: FOLLOW HUMAN TRANSPORT")
print("=" * 70)


transport_z = (
    eef_start[2]
    + TRANSPORT_LIFT_EXTRA
)


log_rows = []


for i, offset_xy in enumerate(
    transformed_xy
):

    target = eef_start.copy()

    target[0] += offset_xy[0]
    target[1] += offset_xy[1]

    target[2] = transport_z

    obs = move_to(
        env,
        obs,
        target,
        f"Waypoint {i:02d}"
    )


    bowl_now = obs[
        "akita_black_bowl_1_pos"
    ].copy()

    eef_now = obs[
        "robot0_eef_pos"
    ].copy()


    log_rows.append({
        "waypoint": i,

        "target_x": target[0],
        "target_y": target[1],
        "target_z": target[2],

        "eef_x": eef_now[0],
        "eef_y": eef_now[1],
        "eef_z": eef_now[2],

        "bowl_x": bowl_now[0],
        "bowl_y": bowl_now[1],
        "bowl_z": bowl_now[2],
    })


    save_image(
        obs,
        f"transport_{i:02d}.png"
    )

# ============================================================
# STAGE 4
# MOVE ABOVE PLATE CENTRE
# ============================================================

print()
print("=" * 70)
print("STAGE 4: ALIGN ABOVE PLATE")
print("=" * 70)

# Current EEF / bowl state
eef_now = obs[
    "robot0_eef_pos"
].copy()

bowl_now = obs[
    "akita_black_bowl_1_pos"
].copy()

plate_now = obs[
    "plate_1_pos"
].copy()


# Bowl is not exactly under the EEF.
# Preserve the current EEF-to-bowl XY grasp offset.
eef_to_bowl_xy = (
    eef_now[:2]
    - bowl_now[:2]
)

print(
    "EEF-to-bowl XY offset:",
    np.round(
        eef_to_bowl_xy,
        4
    )
)


# To put the bowl centre over the plate centre,
# shift the EEF by the same relationship.
target_above_plate = eef_now.copy()

target_above_plate[0] = (
    plate_now[0]
    + eef_to_bowl_xy[0]
)

target_above_plate[1] = (
    plate_now[1]
    + eef_to_bowl_xy[1]
)

# Keep current safe transport height
target_above_plate[2] = eef_now[2]


obs = move_to(
    env,
    obs,
    target_above_plate,
    "Align bowl over plate"
)


save_image(
    obs,
    "20_aligned_above_plate.png"
)


# ============================================================
# STAGE 5
# DESCEND TOWARD PLATE
# ============================================================

print()
print("=" * 70)
print("STAGE 5: DESCEND TO PLATE")
print("=" * 70)


# Re-read current geometry
eef_now = obs[
    "robot0_eef_pos"
].copy()

bowl_now = obs[
    "akita_black_bowl_1_pos"
].copy()

plate_now = obs[
    "plate_1_pos"
].copy()


# Current vertical relationship between EEF and bowl
eef_to_bowl_z = (
    eef_now[2]
    - bowl_now[2]
)

print(
    "EEF-to-bowl Z offset:",
    round(
        eef_to_bowl_z,
        4
    )
)


# We want the bowl to descend close to the plate surface.
#
# Keep a small safety clearance first.
PLACE_CLEARANCE = 0.018

desired_bowl_z = (
    plate_now[2]
    + PLACE_CLEARANCE
)

target_place = eef_now.copy()

target_place[2] = (
    desired_bowl_z
    + eef_to_bowl_z
)


print(
    "Plate Z:",
    round(
        plate_now[2],
        4
    )
)

print(
    "Desired bowl Z:",
    round(
        desired_bowl_z,
        4
    )
)

print(
    "EEF place target Z:",
    round(
        target_place[2],
        4
    )
)


obs = move_to(
    env,
    obs,
    target_place,
    "Descend bowl toward plate"
)


save_image(
    obs,
    "21_before_release.png"
)


# ============================================================
# STAGE 6
# OPEN GRIPPER
# ============================================================

print()
print("=" * 70)
print("STAGE 6: RELEASE")
print("=" * 70)


hold_target = obs[
    "robot0_eef_pos"
].copy()


for _ in range(40):

    current = obs[
        "robot0_eef_pos"
    ].copy()

    error = (
        hold_target
        - current
    )

    command = np.clip(
        KP * error,
        -0.08,
        0.08
    )

    action = np.array([
        command[0],
        command[1],
        command[2],

        0.0,
        0.0,
        0.0,

        -1.0,  # open gripper
    ])

    obs, reward, done, info = (
        env.step(action)
    )


save_image(
    obs,
    "22_released.png"
)


# ============================================================
# STAGE 7
# RETRACT
# ============================================================

print()
print("=" * 70)
print("STAGE 7: RETRACT")
print("=" * 70)


eef_now = obs[
    "robot0_eef_pos"
].copy()

retract_target = (
    eef_now.copy()
)

retract_target[2] += 0.08


# Separate move loop because move_to() currently keeps
# the gripper CLOSED.
for step in range(100):

    current = obs[
        "robot0_eef_pos"
    ].copy()

    error = (
        retract_target
        - current
    )

    if np.linalg.norm(
        error
    ) < POSITION_TOLERANCE:

        break

    command = np.clip(
        KP * error,
        -MAX_ACTION,
        MAX_ACTION
    )

    action = np.array([
        command[0],
        command[1],
        command[2],

        0.0,
        0.0,
        0.0,

        -1.0,  # remain open
    ])

    obs, reward, done, info = (
        env.step(action)
    )


save_image(
    obs,
    "23_final.png"
)


# ============================================================
# FINAL TASK METRICS
# ============================================================

final_bowl = obs[
    "akita_black_bowl_1_pos"
].copy()

final_plate = obs[
    "plate_1_pos"
].copy()


final_xy_distance = (
    np.linalg.norm(
        final_bowl[:2]
        - final_plate[:2]
    )
)

final_z_difference = (
    abs(
        final_bowl[2]
        - final_plate[2]
    )
)


print()
print("=" * 70)
print("FINAL PICK-AND-PLACE RESULT")
print("=" * 70)

print(
    "Final bowl:",
    np.round(
        final_bowl,
        4
    )
)

print(
    "Plate:",
    np.round(
        final_plate,
        4
    )
)

print(
    "Bowl-to-plate XY distance:",
    round(
        final_xy_distance,
        4
    ),
    "m"
)

print(
    "Bowl/plate Z difference:",
    round(
        final_z_difference,
        4
    ),
    "m"
)

print(
    "Environment reward:",
    reward
)

print(
    "Environment done:",
    done
)

# ============================================================
# FINAL TRANSPORT REPORT
# ============================================================

final_bowl = obs[
    "akita_black_bowl_1_pos"
].copy()

final_eef = obs[
    "robot0_eef_pos"
].copy()


bowl_to_plate_xy = (
    np.linalg.norm(
        final_bowl[:2]
        - plate_pos[:2]
    )
)


print()
print("=" * 70)
print("TRANSPORT RESULT")
print("=" * 70)

print(
    "Final EEF:",
    np.round(
        final_eef,
        4
    )
)

print(
    "Final bowl:",
    np.round(
        final_bowl,
        4
    )
)

print(
    "Plate:",
    np.round(
        plate_pos,
        4
    )
)

print(
    "Final bowl-to-plate XY distance:",
    round(
        bowl_to_plate_xy,
        4
    ),
    "m"
)


pd.DataFrame(
    log_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "transport_log.csv"
    ),
    index=False
)


save_image(
    obs,
    "99_transport_final.png"
)


env.close()
