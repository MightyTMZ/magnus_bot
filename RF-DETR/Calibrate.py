import cv2
import json
import os

from board import CROP_PATH, ROTATE_CODE, prep_frame, save_crop

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Step 1: crop ---------------------------------------------------------
# Grab one still (rotated the same way detection will rotate) and let the user
# drag a box around the board. ENTER/SPACE accepts; press 'c' (empty box) to
# clear any crop and use the full rotated frame.
ret, raw = cap.read()
if not ret:
    print("No frame from camera 0. Is iVCam running?")
    cap.release()
    raise SystemExit(1)

rotated = raw if ROTATE_CODE is None else cv2.rotate(raw, ROTATE_CODE)
roi = cv2.selectROI("drag a box around the board, then ENTER (empty box = no crop)", rotated, showCrosshair=False)
cv2.destroyAllWindows()

x, y, w, h = roi
if w > 0 and h > 0:
    save_crop(x, y, w, h)
    print(f"Crop saved to crop.json: {roi}")
else:
    if os.path.exists(CROP_PATH):
        os.remove(CROP_PATH)
    print("No crop (using full rotated frame).")

# --- Step 2: corners ------------------------------------------------------
# Click the four physical board corners IN ORDER: a8 -> h8 -> h1 -> a1.
corners = []
printed = False


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
        corners.append([x, y])
        labels = ["a8", "h8", "h1", "a1"]
        print(f"Added corner {len(corners)} ({labels[len(corners) - 1]}): {x}, {y}")


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', on_click)

while True:
    ret, raw = cap.read()
    if not ret:
        break
    frame = prep_frame(raw)
    if not printed:
        fh, fw = frame.shape[:2]
        print(f"Calibrate frame size (rotated+cropped): {fw}x{fh}")
        print("Click corners in order: a8 -> h8 -> h1 -> a1")
        printed = True
    for (cx, cy) in corners:
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

    if len(corners) == 4:
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(corners) == 4:
    with open('corners.json', 'w') as f:
        json.dump(corners, f)
    print(f"Corners saved to corners.json: {corners}")
else:
    print("Calibration cancelled, need exactly 4 corners.")
