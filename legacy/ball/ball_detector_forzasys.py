import os
import csv

import cv2
import numpy as np

from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

MODEL_PATH = "models/yolov8m_forzasys_soccer.pt"

OUTPUT_VIDEO = "outputs/ball_detector_forzasys.mp4"
OUTPUT_CSV = "outputs/ball_detector_forzasys.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Model
# ============================================================

DEVICE = "mps"

model = YOLO(
    MODEL_PATH
)


# ============================================================
# Model Classes
#
# {0: player, 1: ball, 2: logo}
# ============================================================

BALL_CLASS_ID = 1


# ============================================================
# Detection Parameters
# ============================================================

# 先保持比較低，讓我們看 detector 本身的行為。
# 暫時不要用 temporal filtering。
CONF_THRESHOLD = 0.10

IMGSZ = 1280


# ============================================================
# Video
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


writer = cv2.VideoWriter(

    OUTPUT_VIDEO,

    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),

    fps,

    (
        width,
        height
    )
)


if not writer.isOpened():

    raise RuntimeError(
        f"Cannot create video: {OUTPUT_VIDEO}"
    )


# ============================================================
# Statistics
# ============================================================

frame_idx = 0

frames_with_ball = 0

frames_with_one_candidate = 0

frames_with_multiple_candidates = 0

total_candidates = 0

confidence_values = []

rows = []


print("")
print(
    "========================================"
)

print(
    "FORZASYS BALL DETECTOR"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    total_frames
)

print(
    "FPS:",
    fps
)

print(
    "Resolution:",
    f"{width}x{height}"
)

print("")

print(
    "Ball class ID:",
    BALL_CLASS_ID
)

print(
    "Confidence threshold:",
    CONF_THRESHOLD
)


# ============================================================
# Main Loop
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    annotated = frame.copy()


    # ========================================================
    # Ball-only inference
    # ========================================================

    result = model(

        frame,

        imgsz=IMGSZ,

        conf=CONF_THRESHOLD,

        classes=[
            BALL_CLASS_ID
        ],

        device=DEVICE,

        verbose=False

    )[0]


    boxes = result.boxes


    candidates = []


    # ========================================================
    # Read detections
    # ========================================================

    if (
        boxes is not None
        and
        len(boxes) > 0
    ):

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )


        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )


        classes = (
            boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )


        for box, confidence, class_id in zip(
            xyxy,
            confidences,
            classes
        ):

            # Safety check
            if (
                class_id
                != BALL_CLASS_ID
            ):

                continue


            x1, y1, x2, y2 = box


            center_x = float(
                (
                    x1 + x2
                )
                /
                2.0
            )

            center_y = float(
                (
                    y1 + y2
                )
                /
                2.0
            )


            candidate = {

                "box":
                    box,

                "confidence":
                    float(
                        confidence
                    ),

                "center_x":
                    center_x,

                "center_y":
                    center_y
            }


            candidates.append(
                candidate
            )


    # ========================================================
    # Statistics
    # ========================================================

    candidate_count = len(
        candidates
    )


    total_candidates += (
        candidate_count
    )


    if candidate_count > 0:

        frames_with_ball += 1


    if candidate_count == 1:

        frames_with_one_candidate += 1


    elif candidate_count > 1:

        frames_with_multiple_candidates += 1


    # ========================================================
    # Draw candidates
    #
    # IMPORTANT:
    # We draw EVERY candidate.
    #
    # No highest-confidence selection.
    # No temporal association.
    #
    # We want to see the raw detector behaviour.
    # ========================================================

    for candidate_index, candidate in enumerate(
        candidates
    ):

        x1, y1, x2, y2 = (
            candidate[
                "box"
            ]
        )


        confidence = (
            candidate[
                "confidence"
            ]
        )


        center_x = (
            candidate[
                "center_x"
            ]
        )

        center_y = (
            candidate[
                "center_y"
            ]
        )


        confidence_values.append(
            confidence
        )


        # ====================================================
        # Confidence visualization
        #
        # >= 0.50  green
        # >= 0.25  yellow
        # <  0.25  red
        # ====================================================

        if confidence >= 0.50:

            color = (
                0,
                255,
                0
            )

        elif confidence >= 0.25:

            color = (
                0,
                255,
                255
            )

        else:

            color = (
                0,
                0,
                255
            )


        cv2.rectangle(

            annotated,

            (
                int(x1),
                int(y1)
            ),

            (
                int(x2),
                int(y2)
            ),

            color,

            2
        )


        cv2.circle(

            annotated,

            (
                int(center_x),
                int(center_y)
            ),

            6,

            color,

            -1
        )


        cv2.putText(

            annotated,

            (
                f"BALL "
                f"{confidence:.2f}"
            ),

            (
                int(x1),
                max(
                    int(y1) - 7,
                    20
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            color,

            2,

            cv2.LINE_AA
        )


        # ====================================================
        # CSV
        # ====================================================

        rows.append(
            {
                "frame":
                    frame_idx,

                "time_sec":
                    frame_idx
                    /
                    fps,

                "candidate_index":
                    candidate_index,

                "confidence":
                    confidence,

                "x1":
                    float(x1),

                "y1":
                    float(y1),

                "x2":
                    float(x2),

                "y2":
                    float(y2),

                "center_x":
                    center_x,

                "center_y":
                    center_y
            }
        )


    # ========================================================
    # Overlay
    # ========================================================

    cv2.putText(

        annotated,

        (
            f"ForzaSys Ball Detector | "
            f"Candidates={candidate_count}"
        ),

        (
            20,
            35
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.70,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    if candidate_count > 0:

        best_confidence = max(

            candidate[
                "confidence"
            ]

            for candidate
            in candidates
        )


        cv2.putText(

            annotated,

            (
                f"Best conf="
                f"{best_confidence:.2f}"
            ),

            (
                20,
                65
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (
                255,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # Debug
    # ========================================================

    if frame_idx % 30 == 0:

        if candidate_count > 0:

            best_conf = max(

                candidate[
                    "confidence"
                ]

                for candidate
                in candidates
            )


            conf_text = (
                f"{best_conf:.2f}"
            )

        else:

            conf_text = "N/A"


        print(

            f"frame={frame_idx:04d} | "

            f"candidates="
            f"{candidate_count} | "

            f"best_conf="
            f"{conf_text}"
        )


    writer.write(
        annotated
    )


    frame_idx += 1


# ============================================================
# Finish
# ============================================================

cap.release()

writer.release()


# ============================================================
# Save CSV
# ============================================================

columns = [

    "frame",
    "time_sec",

    "candidate_index",

    "confidence",

    "x1",
    "y1",
    "x2",
    "y2",

    "center_x",
    "center_y"
]


with open(

    OUTPUT_CSV,

    "w",

    newline="",

    encoding="utf-8"

) as f:

    csv_writer = csv.DictWriter(

        f,

        fieldnames=columns
    )


    csv_writer.writeheader()


    csv_writer.writerows(
        rows
    )


# ============================================================
# Summary
# ============================================================

detection_rate = (

    frames_with_ball
    /
    frame_idx
    *
    100.0

    if frame_idx > 0

    else 0.0
)


multiple_rate = (

    frames_with_multiple_candidates
    /
    frame_idx
    *
    100.0

    if frame_idx > 0

    else 0.0
)


print("")
print(
    "========================================"
)

print(
    "FORZASYS DETECTOR SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    frame_idx
)

print(
    "Frames with >=1 ball candidate:",
    frames_with_ball
)

print(
    "Raw detection coverage:",
    f"{detection_rate:.1f}%"
)

print("")

print(
    "Frames with exactly 1 candidate:",
    frames_with_one_candidate
)

print(
    "Frames with multiple candidates:",
    frames_with_multiple_candidates
)

print(
    "Multiple-candidate rate:",
    f"{multiple_rate:.1f}%"
)

print("")

print(
    "Total ball candidates:",
    total_candidates
)


if confidence_values:

    conf = np.asarray(
        confidence_values
    )


    print("")

    print(
        "Confidence median:",
        f"{np.median(conf):.3f}"
    )

    print(
        "Confidence P25:",
        f"{np.percentile(conf, 25):.3f}"
    )

    print(
        "Confidence P75:",
        f"{np.percentile(conf, 75):.3f}"
    )

    print(
        "Confidence max:",
        f"{np.max(conf):.3f}"
    )


print("")

print(
    "Video:",
    OUTPUT_VIDEO
)

print(
    "CSV:",
    OUTPUT_CSV
)
