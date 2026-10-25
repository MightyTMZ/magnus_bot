"""Local RF-DETR piece detector, shared by vision_server.py.

Mirrors the model setup in Chessdetect.py (same weights, class list and threshold). If the
weights or class list change, update both until Chessdetect.py imports from here.
"""

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "model", "weights_2.0.pt")

# Class ids from the Roboflow COCO export of chess-full-kntuv/1.
# Id 0 is Roboflow's superclass; "bishop" is a stray label with one annotation.
CLASS_NAMES = [
    "pieces",
    "bishop",
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
]

CONFIDENCE = 0.35


def load_model(weights=WEIGHTS):
    from rfdetr import RFDETRSmall

    print(f"Loading RF-DETR Small weights from {weights}")
    model = RFDETRSmall(pretrain_weights=weights)
    try:
        model.optimize_for_inference()
    except Exception as e:  # optional speed-up; plain inference still works
        print(f"optimize_for_inference skipped: {e}")
    return model


def detect(model, frame, threshold=CONFIDENCE):
    """Run the model on a BGR OpenCV frame; detections carry a "class_name" array."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = model.predict(rgb, threshold=threshold)
    dets.data["class_name"] = np.array(
        [CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else str(i) for i in dets.class_id]
    )
    return dets
