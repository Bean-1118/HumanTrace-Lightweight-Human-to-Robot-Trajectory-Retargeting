import os
import glob
import cv2
import imageio.v2 as imageio


INPUT_DIR = "results/final_pick_place"
OUTPUT_GIF = "results/humantrace_pick_place.gif"

frames = []

# ============================================================
# Build frame order
# ============================================================

frame_paths = []

# 1) Initial scene
frame_paths.append(
    os.path.join(INPUT_DIR, "00_start.png")
)

# 2) Continuous grasp / lift frames
grasp_frames = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "grasp_*.png")
    )
)

frame_paths.extend(grasp_frames)

# 3) After-grasp checkpoint
frame_paths.append(
    os.path.join(INPUT_DIR, "01_after_grasp.png")
)

# 4) Human transport
for i in range(20):
    frame_paths.append(
        os.path.join(INPUT_DIR, f"transport_{i:02d}.png")
    )

# 5) Placement / release / final
frame_paths.extend([
    os.path.join(INPUT_DIR, "20_aligned_above_plate.png"),
    os.path.join(INPUT_DIR, "21_before_release.png"),
    os.path.join(INPUT_DIR, "22_released.png"),
    os.path.join(INPUT_DIR, "23_final.png"),
])

# ============================================================
# Load images
# ============================================================

for path in frame_paths:
    if not os.path.exists(path):
        print("Missing:", path)
        continue

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frames.append(image)

# ============================================================
# Save GIF
# ============================================================

imageio.mimsave(
    OUTPUT_GIF,
    frames,
    duration=0.12,
    loop=0
)

print("Saved:", OUTPUT_GIF)
print("Total frames:", len(frames))
print("Grasp frames:", len(grasp_frames))
