import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HUMAN_LOG = "results/final_pick_place/transport_log.csv"
STRAIGHT_LOG = "results/final_pick_place_straight/transport_log.csv"

OUTPUT_CSV = "results/comparison_metrics.csv"
OUTPUT_PLOT = "results/humantrace_vs_straight.png"


def path_length(xy):
    diffs = np.diff(xy, axis=0)
    return np.linalg.norm(diffs, axis=1).sum()


def tracking_errors(df):
    target = df[
        ["target_x", "target_y", "target_z"]
    ].to_numpy()

    actual = df[
        ["eef_x", "eef_y", "eef_z"]
    ].to_numpy()

    return np.linalg.norm(
        target - actual,
        axis=1
    )


def max_lateral_deviation(xy):
    start = xy[0]
    end = xy[-1]

    line = end - start
    line_length = np.linalg.norm(line)

    if line_length < 1e-9:
        return 0.0

    deviations = []

    for point in xy:
        rel = point - start

        deviation = abs(
            line[0] * rel[1]
            - line[1] * rel[0]
        ) / line_length

        deviations.append(deviation)

    return max(deviations)


# ============================================================
# LOAD DATA
# ============================================================

human = pd.read_csv(HUMAN_LOG)
straight = pd.read_csv(STRAIGHT_LOG)


human_target_xy = human[
    ["target_x", "target_y"]
].to_numpy()

straight_target_xy = straight[
    ["target_x", "target_y"]
].to_numpy()

human_bowl_xy = human[
    ["bowl_x", "bowl_y"]
].to_numpy()

straight_bowl_xy = straight[
    ["bowl_x", "bowl_y"]
].to_numpy()


# ============================================================
# METRICS
# ============================================================

human_tracking = tracking_errors(human)
straight_tracking = tracking_errors(straight)

human_target_length = path_length(
    human_target_xy
)

straight_target_length = path_length(
    straight_target_xy
)

human_bowl_length = path_length(
    human_bowl_xy
)

straight_bowl_length = path_length(
    straight_bowl_xy
)

human_lateral_deviation = (
    max_lateral_deviation(
        human_target_xy
    )
)

straight_lateral_deviation = (
    max_lateral_deviation(
        straight_target_xy
    )
)


# Values measured from final successful runs
human_final_placement_error = 0.0054
straight_final_placement_error = 0.0055

human_pre_alignment_error = 0.0160
straight_pre_alignment_error = 0.0259


metrics = pd.DataFrame([
    {
        "method": "HumanTrace",
        "transport_waypoints": len(human),
        "target_path_length_m": human_target_length,
        "bowl_path_length_m": human_bowl_length,
        "mean_tracking_error_m": human_tracking.mean(),
        "max_tracking_error_m": human_tracking.max(),
        "max_lateral_deviation_m": human_lateral_deviation,
        "pre_alignment_error_m": human_pre_alignment_error,
        "final_placement_error_m": human_final_placement_error,
        "task_success": 1,
    },

    {
        "method": "Straight Line",
        "transport_waypoints": len(straight),
        "target_path_length_m": straight_target_length,
        "bowl_path_length_m": straight_bowl_length,
        "mean_tracking_error_m": straight_tracking.mean(),
        "max_tracking_error_m": straight_tracking.max(),
        "max_lateral_deviation_m": straight_lateral_deviation,
        "pre_alignment_error_m": straight_pre_alignment_error,
        "final_placement_error_m": straight_final_placement_error,
        "task_success": 1,
    },
])


metrics.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 90)
print("HUMANTRACE VS STRAIGHT-LINE BASELINE")
print("=" * 90)

print(
    metrics.to_string(
        index=False
    )
)

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    "HumanTrace target path length:",
    round(human_target_length, 4),
    "m"
)

print(
    "Straight target path length:",
    round(straight_target_length, 4),
    "m"
)

print(
    "HumanTrace max lateral deviation:",
    round(human_lateral_deviation, 4),
    "m"
)

print(
    "Straight max lateral deviation:",
    round(straight_lateral_deviation, 4),
    "m"
)

print(
    "HumanTrace mean tracking error:",
    round(human_tracking.mean(), 4),
    "m"
)

print(
    "Straight mean tracking error:",
    round(straight_tracking.mean(), 4),
    "m"
)

print(
    "HumanTrace final placement error:",
    human_final_placement_error,
    "m"
)

print(
    "Straight final placement error:",
    straight_final_placement_error,
    "m"
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 7))

plt.plot(
    human_target_xy[:, 0],
    human_target_xy[:, 1],
    marker="o",
    label="HumanTrace target"
)

plt.plot(
    straight_target_xy[:, 0],
    straight_target_xy[:, 1],
    marker="o",
    label="Straight-line target"
)

plt.plot(
    human_bowl_xy[:, 0],
    human_bowl_xy[:, 1],
    linestyle="--",
    label="HumanTrace bowl"
)

plt.plot(
    straight_bowl_xy[:, 0],
    straight_bowl_xy[:, 1],
    linestyle="--",
    label="Straight-line bowl"
)

plt.scatter(
    human_target_xy[0, 0],
    human_target_xy[0, 1],
    s=100,
    label="Transport start"
)

plt.scatter(
    human_target_xy[-1, 0],
    human_target_xy[-1, 1],
    s=100,
    label="Transport end"
)

plt.xlabel("Robot X position (m)")
plt.ylabel("Robot Y position (m)")
plt.title("HumanTrace vs Straight-Line Transport")

plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=200
)

plt.close()


print()
print("Saved metrics:", OUTPUT_CSV)
print("Saved plot:", OUTPUT_PLOT)
