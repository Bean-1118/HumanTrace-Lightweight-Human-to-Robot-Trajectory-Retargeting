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
    "results/final_pick_place_straight"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

GRIPPER_CLOSE = +1.0
GRIPPER_OPEN = -1.0

KP = 6.0
MAX_ACTION = 0.18

POSITION_TOLERANCE = 0.008
MAX_STEPS_PER_WAYPOINT = 80

PLACE_CLEARANCE = 0.018


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
# MOVE EEF TO XYZ TARGET
# ============================================================

def move_to(
    env,
    obs,
    target,
    name,
    gripper_value=GRIPPER_CLOSE
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

            0.0,
            0.0,
            0.0,

            gripper_value,
        ])

        obs, reward, done, info = (
            env.step(action)
        )

    final_error = np.linalg.norm(
        target
        - obs["robot0_eef_pos"]
    )

    print(
        name,
        "| target:",
        np.round(
            target,
            4
        ),
        "| actual:",
        np.round(
            obs["robot0_eef_pos"],
            4
        ),
        "| error:",
        round(
            final_error,
            4
        ),
        "| reached:",
        reached
    )

    return obs


# ============================================================
# LOAD SMALL GRASP PRIMITIVE
# ============================================================

primitive = np.load(
    GRASP_PRIMITIVE_PATH
)

initial_state = primitive[
    "initial_state"
]

grasp_actions = primitive[
    "grasp_actions"
]


print()
print("=" * 70)
print("GRASP PRIMITIVE")
print("=" * 70)

print(
    "Initial state shape:",
    initial_state.shape
)

print(
    "Grasp actions shape:",
    grasp_actions.shape
)


# ============================================================
# LOAD HUMAN WAYPOINT FILE
#
# We ONLY use this to match the same number of waypoints
# as HumanTrace, so comparison is fair.
# ============================================================

human_df = pd.read_csv(
    WAYPOINT_PATH
)

num_waypoints = len(
    human_df
)


print()
print("=" * 70)
print("STRAIGHT BASELINE CONFIGURATION")
print("=" * 70)

print(
    "Number of transport waypoints:",
    num_waypoints
)


# ============================================================
# CREATE LIBERO ENVIRONMENT
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
# RESTORE SAME INITIAL STATE
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
# SAME GRASP + LIFT PRIMITIVE AS HUMANTRACE
# ============================================================

print()
print("=" * 70)
print("STAGE 1: GRASP + LIFT")
print("=" * 70)


for action in grasp_actions:

    obs, reward, done, info = (
        env.step(action)
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
# STRAIGHT-LINE BASELINE
# ============================================================

print()
print("=" * 70)
print("STAGE 2: STRAIGHT-LINE BASELINE")
print("=" * 70)


robot_vector = (
    plate_pos[:2]
    - bowl_start[:2]
)


alphas = np.linspace(
    0.0,
    1.0,
    num_waypoints
)


transformed_xy = np.array([
    alpha * robot_vector
    for alpha in alphas
])


print(
    "Robot bowl→plate vector:",
    np.round(
        robot_vector,
        4
    )
)

print(
    "Straight endpoint:",
    np.round(
        transformed_xy[-1],
        4
    )
)


# ============================================================
# STAGE 3
# FOLLOW STRAIGHT TRANSPORT
# ============================================================

print()
print("=" * 70)
print("STAGE 3: FOLLOW STRAIGHT TRANSPORT")
print("=" * 70)


transport_z = (
    eef_start[2]
)


log_rows = []


for i, offset_xy in enumerate(
    transformed_xy
):

    target = (
        eef_start.copy()
    )

    target[0] += (
        offset_xy[0]
    )

    target[1] += (
        offset_xy[1]
    )

    target[2] = (
        transport_z
    )


    obs = move_to(
        env,
        obs,
        target,
        f"Waypoint {i:02d}",
        GRIPPER_CLOSE
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
# TRANSPORT RESULT BEFORE PLACEMENT
# ============================================================

transport_final_bowl = obs[
    "akita_black_bowl_1_pos"
].copy()

transport_final_eef = obs[
    "robot0_eef_pos"
].copy()


transport_xy_distance = (
    np.linalg.norm(
        transport_final_bowl[:2]
        - plate_pos[:2]
    )
)


print()
print("=" * 70)
print("TRANSPORT RESULT")
print("=" * 70)

print(
    "Final transport EEF:",
    np.round(
        transport_final_eef,
        4
    )
)

print(
    "Final transport bowl:",
    np.round(
        transport_final_bowl,
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
    "Bowl-to-plate XY distance before alignment:",
    round(
        transport_xy_distance,
        4
    ),
    "m"
)


save_image(
    obs,
    "19_transport_final.png"
)


# ============================================================
# STAGE 4
# ALIGN BOWL ABOVE PLATE
# ============================================================

print()
print("=" * 70)
print("STAGE 4: ALIGN ABOVE PLATE")
print("=" * 70)


eef_now = obs[
    "robot0_eef_pos"
].copy()

bowl_now = obs[
    "akita_black_bowl_1_pos"
].copy()

plate_now = obs[
    "plate_1_pos"
].copy()


# Preserve current EEF-to-bowl grasp offset
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


target_above_plate = (
    eef_now.copy()
)


target_above_plate[0] = (
    plate_now[0]
    + eef_to_bowl_xy[0]
)

target_above_plate[1] = (
    plate_now[1]
    + eef_to_bowl_xy[1]
)

target_above_plate[2] = (
    eef_now[2]
)


obs = move_to(
    env,
    obs,
    target_above_plate,
    "Align bowl over plate",
    GRIPPER_CLOSE
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


eef_now = obs[
    "robot0_eef_pos"
].copy()

bowl_now = obs[
    "akita_black_bowl_1_pos"
].copy()

plate_now = obs[
    "plate_1_pos"
].copy()


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


desired_bowl_z = (
    plate_now[2]
    + PLACE_CLEARANCE
)


target_place = (
    eef_now.copy()
)

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
    "Descend bowl toward plate",
    GRIPPER_CLOSE
)


save_image(
    obs,
    "21_before_release.png"
)


# ============================================================
# STAGE 6
# RELEASE
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

        GRIPPER_OPEN,
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

retract_target[2] += (
    0.08
)


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

        GRIPPER_OPEN,
    ])

    obs, reward, done, info = (
        env.step(action)
    )


save_image(
    obs,
    "23_final.png"
)


# ============================================================
# FINAL METRICS
# ============================================================

final_bowl = obs[
    "akita_black_bowl_1_pos"
].copy()

final_plate = obs[
    "plate_1_pos"
].copy()

final_eef = obs[
    "robot0_eef_pos"
].copy()


final_xy_distance = (
    np.linalg.norm(
        final_bowl[:2]
        - final_plate[:2]
    )
)

final_z_difference = abs(
    final_bowl[2]
    - final_plate[2]
)


print()
print("=" * 70)
print("FINAL PICK-AND-PLACE RESULT")
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
# SAVE TRANSPORT LOG
# ============================================================

pd.DataFrame(
    log_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "transport_log.csv"
    ),
    index=False
)


env.close()
