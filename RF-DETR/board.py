import cv2
import json
import os
import numpy as np

BOARD_SIZE = 800
SQUARE = BOARD_SIZE // 8
# Higher camera: sample farther up the box so the point stays on the piece's square.
FOOT = 0.35

HERE = os.path.dirname(os.path.abspath(__file__))
CROP_PATH = os.path.join(HERE, "crop.json")
# The phone is mounted sideways, so every frame is rotated before anything else.
# Set to None to disable rotation.
ROTATE_CODE = cv2.ROTATE_90_CLOCKWISE


def load_crop(path=CROP_PATH):
    """Return (x, y, w, h) in ROTATED-frame coordinates, or None for no crop."""
    if os.path.exists(path):
        with open(path) as f:
            c = json.load(f)
        if c and c.get("w") and c.get("h"):
            return c["x"], c["y"], c["w"], c["h"]
    return None


def save_crop(x, y, w, h, path=CROP_PATH):
    with open(path, "w") as f:
        json.dump({"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, f)


def prep_frame(frame):
    """Rotate then crop a raw camera frame.

    MUST be applied identically in calibration and detection so corners.json and
    the Roboflow detections live in the same coordinate space.
    """
    if ROTATE_CODE is not None:
        frame = cv2.rotate(frame, ROTATE_CODE)
    crop = load_crop()
    if crop:
        x, y, w, h = crop
        frame = frame[y:y + h, x:x + w]
    return frame

def load_homography(path="corners.json"):
    with open(path) as f:
        corners = json.load(f)
    
    src = np.float32(corners)
    dst = np.float32([[0, 0], [BOARD_SIZE, 0], [BOARD_SIZE, BOARD_SIZE], [0, BOARD_SIZE]])
    return cv2.getPerspectiveTransform(src, dst)

def point_to_square(x,y,H):
    pt = np.float32([[[x,y]]])
    flat = cv2.perspectiveTransform(pt, H)
    fx, fy = flat[0][0]
    col = int(fx // SQUARE)
    row = int(fy // SQUARE)

    if col < 0 or col >= 8 or row < 0 or row >= 8:
        return None
    else:
        return "abcdefgh"[col] + str(8-row)

def box_anchor(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, y2 - FOOT * (y2 - y1)

def box_to_square(box,H):
    x, y = box_anchor(box)
    return point_to_square(x,y,H)

def build_board(detections, H):
    board = {}
    best = {}
    confidences = detections.confidence
    for i, (box, name) in enumerate(zip(detections.xyxy, detections.data["class_name"])):
        square = box_to_square(box, H)
        if square is None:
            continue
        if str(name).startswith("white"):
            color = "W"
        else:
            color = "B"
        score = float(confidences[i]) if confidences is not None else 0.0
        if square not in best or score > best[square]:
            best[square] = score
            board[square] = color
    return board

def print_board(board):
    for rank in range(8, 0, -1):
        row = []
        for file in "abcdefgh":
            square = file + str(rank)
            if square in board:
                row.append(board[square])
            else:
                row.append(".")
        print(f"{rank}  " + " ".join(row))   # ← rank number at the start
    print("   a b c d e f g h")              # ← file letters at the bottom
    print()                                  # ← blank line between prints
def draw_grid(frame, H):
    """Draw the 8x8 board grid onto the camera image to check calibration."""
    H_inv = np.linalg.inv(H)
    for i in range(9):
        p = i * SQUARE
        # vertical line (file boundary) and horizontal line (rank boundary)
        for a, b in [((p, 0), (p, BOARD_SIZE)), ((0, p), (BOARD_SIZE, p))]:
            pts = np.float32([[a, b]])
            cam = cv2.perspectiveTransform(pts, H_inv)[0]
            cv2.line(frame, tuple(map(int, cam[0])), tuple(map(int, cam[1])), (0, 255, 255), 1)

    # label each square so you can read it on screen
    for col in range(8):
        for row in range(8):
            center = np.float32([[[col * SQUARE + SQUARE / 2, row * SQUARE + SQUARE / 2]]])
            cx, cy = cv2.perspectiveTransform(center, H_inv)[0][0]
            name = "abcdefgh"[col] + str(8 - row)
            cv2.putText(frame, name, (int(cx) - 10, int(cy) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return frame


def draw_points(frame, detections, H):
    """Draw the point used for each piece, colored by color, with its square name."""
    for box, name in zip(detections.xyxy, detections.data["class_name"]):
        x, y = box_anchor(box)
        square = point_to_square(x, y, H)
        dot = (255, 255, 255) if name.startswith("white") else (0, 0, 255)
        cv2.circle(frame, (int(x), int(y)), 5, dot, -1)
        cv2.putText(frame, square or "off", (int(x) + 6, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot, 1)
    return frame