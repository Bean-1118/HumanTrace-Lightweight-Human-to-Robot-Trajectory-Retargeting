import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

INPUT = "data/processed/demo03_points.csv"

OUTPUT_FULL = "data/processed/demo03_clean.csv"
OUTPUT_WAYPOINTS = "data/processed/demo03_waypoints.csv"

# --------------------------------------------------
# 1. Load manually labelled trajectory
# --------------------------------------------------

df = pd.read_csv(INPUT)

print("Raw points:", len(df))

# --------------------------------------------------
# 2. Keep only the transport phase
#
# Frame 0-7   : grasping
# Frame 27-74 : actual object transport
# Frame 76+   : release / hand adjustment
# --------------------------------------------------

START_FRAME = 92
END_FRAME = 347

df = df[
    (df["frame"] >= START_FRAME) &
    (df["frame"] <= END_FRAME)
].copy()

# If multiple clicks accidentally occurred on same frame,
# average them.
df = (
    df.groupby("frame", as_index=False)
      .agg({"x": "mean", "y": "mean"})
)

print("Transport points:", len(df))

x_raw = df["x"].to_numpy(dtype=float)
y_raw = df["y"].to_numpy(dtype=float)

# --------------------------------------------------
# 3. Smooth the trajectory
# --------------------------------------------------

# Must be odd and smaller than number of points
window = 9

x_smooth = savgol_filter(
    x_raw,
    window_length=window,
    polyorder=2
)

y_smooth = savgol_filter(
    y_raw,
    window_length=window,
    polyorder=2
)

# --------------------------------------------------
# 4. Convert to relative trajectory
#
# The first transport point becomes (0, 0)
# --------------------------------------------------

dx = x_smooth - x_smooth[0]
dy = y_smooth - y_smooth[0]

# Image coordinates:
# +y = downward
#
# Convert to normal Cartesian convention:
# +y = upward
dy = -dy

# --------------------------------------------------
# 5. Normalize while preserving trajectory shape
# --------------------------------------------------

distance_from_start = np.sqrt(
    dx ** 2 + dy ** 2
)

scale = np.max(distance_from_start)

dx_norm = dx / scale
dy_norm = dy / scale

# --------------------------------------------------
# 6. Save full smoothed trajectory
# --------------------------------------------------

clean = pd.DataFrame({
    "frame": df["frame"],
    "x_raw": x_raw,
    "y_raw": y_raw,
    "x_smooth": x_smooth,
    "y_smooth": y_smooth,
    "dx_norm": dx_norm,
    "dy_norm": dy_norm,
})

clean.to_csv(
    OUTPUT_FULL,
    index=False
)

# --------------------------------------------------
# 7. Resample into 20 waypoints based on path length
# --------------------------------------------------

points = np.column_stack([
    dx_norm,
    dy_norm
])

segment_lengths = np.sqrt(
    np.sum(
        np.diff(points, axis=0) ** 2,
        axis=1
    )
)

arc_length = np.concatenate([
    [0],
    np.cumsum(segment_lengths)
])

arc_length = arc_length / arc_length[-1]

new_arc = np.linspace(
    0,
    1,
    40
)

waypoint_x = np.interp(
    new_arc,
    arc_length,
    dx_norm
)

waypoint_y = np.interp(
    new_arc,
    arc_length,
    dy_norm
)

waypoints = pd.DataFrame({
    "waypoint": np.arange(40),
    "dx_norm": waypoint_x,
    "dy_norm": waypoint_y,
})

waypoints.to_csv(
    OUTPUT_WAYPOINTS,
    index=False
)

# --------------------------------------------------
# 8. Plot raw vs smoothed trajectory
# --------------------------------------------------

plt.figure(figsize=(7, 7))

# Reverse image y-axis for visual comparison
plt.plot(
    x_raw,
    -y_raw,
    "o--",
    alpha=0.5,
    label="Raw manual labels"
)

plt.plot(
    x_smooth,
    -y_smooth,
    linewidth=3,
    label="Smoothed trajectory"
)

plt.scatter(
    x_smooth[0],
    -y_smooth[0],
    s=100,
    label="Start"
)

plt.scatter(
    x_smooth[-1],
    -y_smooth[-1],
    s=100,
    label="End"
)

plt.xlabel("Image X")
plt.ylabel("Image Y (flipped)")
plt.title("Human Demonstration Trajectory")
plt.axis("equal")
plt.grid()
plt.legend()

plt.savefig(
    "results/demo03_raw_vs_smooth.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 9. Plot normalized robot-ready trajectory
# --------------------------------------------------

plt.figure(figsize=(7, 7))

plt.plot(
    waypoint_x,
    waypoint_y,
    "o-"
)

plt.scatter(
    waypoint_x[0],
    waypoint_y[0],
    s=120,
    label="Start"
)

plt.scatter(
    waypoint_x[-1],
    waypoint_y[-1],
    s=120,
    label="End"
)

plt.xlabel("Normalized X")
plt.ylabel("Normalized Y")
plt.title("Robot-Ready Human Trajectory")
plt.axis("equal")
plt.grid()
plt.legend()

plt.savefig(
    "results/demo03_normalized.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print()
print("Saved:")
print(OUTPUT_FULL)
print(OUTPUT_WAYPOINTS)

print()
print("20 robot-ready waypoints:")
print(waypoints)
