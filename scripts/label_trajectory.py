import cv2
import pandas as pd

VIDEO_PATH = "data/raw/demo01.mp4"
OUTPUT_PATH = "data/processed/demo01_points.csv"

cap = cv2.VideoCapture(VIDEO_PATH)

points = []
frame_id = 0
current_frame = None


def mouse_callback(event, x, y, flags, param):
    global frame_id

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([frame_id, x, y])

        print(
            f"Point {len(points)} | "
            f"frame={frame_id}, x={x}, y={y}"
        )


cv2.namedWindow("Human Demo")
cv2.setMouseCallback("Human Demo", mouse_callback)

# Read first frame
ret, current_frame = cap.read()

if not ret:
    raise RuntimeError("Cannot read video.")

while True:

    display = current_frame.copy()

    # Draw points already selected
    for _, px, py in points:
        cv2.circle(
            display,
            (px, py),
            7,
            (0, 0, 255),
            -1
        )

    cv2.putText(
        display,
        f"Frame: {frame_id}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "D: +5 frames | A: -5 | F: +1 | S: -1 | Click: mark | U: undo | Q: save",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("Human Demo", display)

    key = cv2.waitKey(0) & 0xFF

    # +5 frames
    if key == ord("d"):
        target = frame_id + 5
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

        ret, current_frame = cap.read()

        if ret:
            frame_id = target

    # -5 frames
    elif key == ord("a"):
        target = max(0, frame_id - 5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

        ret, current_frame = cap.read()

        if ret:
            frame_id = target

    # +1 frame
    elif key == ord("f"):
        target = frame_id + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

        ret, current_frame = cap.read()

        if ret:
            frame_id = target

    # -1 frame
    elif key == ord("s"):
        target = max(0, frame_id - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

        ret, current_frame = cap.read()

        if ret:
            frame_id = target

    # Undo last point
    elif key == ord("u"):
        if points:
            removed = points.pop()
            print("Removed:", removed)

    # Save and quit
    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(
    points,
    columns=["frame", "x", "y"]
)

df.to_csv(OUTPUT_PATH, index=False)

print()
print("Saved:")
print(OUTPUT_PATH)
print(df)
