import os
import csv
from collections import deque

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch
)

from sports.configs.soccer import (
    SoccerPitchConfiguration
)


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

BALL_MODEL_PATH = "models/football-ball-detection.pt"
PITCH_MODEL_PATH = "models/football-pitch-detection.pt"

OUTPUT_VIDEO = "outputs/ball_tracking_v1.mp4"
OUTPUT_CSV = "outputs/ball_tracking_v1.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Device
# ============================================================

DEVICE = "mps"


# ============================================================
# Models
# ============================================================

ball_model = YOLO(
    BALL_MODEL_PATH
)

pitch_model = YOLO(
    PITCH_MODEL_PATH
)

CONFIG = SoccerPitchConfiguration()


# ============================================================
# Ball Parameters
# ============================================================

# 球非常小，所以 threshold 先放低
BALL_CONF = 0.10

BALL_IMGSZ = 1280


# ============================================================
# Homography Parameters
#
# 跟 radar_v4 一樣：
# 不做 keypoint EMA
# ============================================================

KP_CONF_THRESHOLD = 0.60

MIN_KEYPOINTS = 6

MIN_INLIER_RATIO = 0.60

RANSAC_THRESHOLD = 250.0

MAX_MEAN_REPROJECTION_ERROR = 30.0

MAX_MEDIAN_REPROJECTION_ERROR = 20.0

MAX_FALLBACK_FRAMES = 3


# ============================================================
# Pitch Geometry
# ============================================================

pitch_vertices = np.asarray(
    CONFIG.vertices,
    dtype=np.float32
)


PITCH_X_MIN = float(
    np.min(
        pitch_vertices[:, 0]
    )
)

PITCH_X_MAX = float(
    np.max(
        pitch_vertices[:, 0]
    )
)

PITCH_Y_MIN = float(
    np.min(
        pitch_vertices[:, 1]
    )
)

PITCH_Y_MAX = float(
    np.max(
        pitch_vertices[:, 1]
    )
)


# ============================================================
# Helper:
# Estimate Homography
# ============================================================

def estimate_homography(
    frame,
    last_H,
    fallback_streak
):

    result = pitch_model(
        frame,
        device=DEVICE,
        verbose=False
    )[0]


    keypoints = (
        sv.KeyPoints.from_ultralytics(
            result
        )
    )


    H = None

    valid_kp_count = 0

    inlier_count = 0

    mean_error = None

    median_error = None

    used_fallback = False


    # ========================================================
    # Current-frame raw keypoints
    # ========================================================

    if len(keypoints.xy) > 0:

        pts = keypoints.xy[0]


        if keypoints.confidence is not None:

            conf = (
                keypoints.confidence[0]
            )

        else:

            conf = np.ones(
                len(pts),
                dtype=np.float32
            )


        valid = (

            (pts[:, 0] > 1)

            &

            (pts[:, 1] > 1)

            &

            (
                conf
                >= KP_CONF_THRESHOLD
            )
        )


        source = (
            pts[valid]
            .astype(np.float32)
        )


        target = (
            pitch_vertices[valid]
            .astype(np.float32)
        )


        valid_kp_count = len(
            source
        )


        # ====================================================
        # RANSAC Homography
        # ====================================================

        if (
            valid_kp_count
            >= MIN_KEYPOINTS
        ):

            H_candidate, inliers = (
                cv2.findHomography(
                    source,
                    target,
                    cv2.RANSAC,
                    RANSAC_THRESHOLD
                )
            )


            if (
                H_candidate is not None
                and
                inliers is not None
            ):

                inlier_mask = (
                    inliers
                    .reshape(-1)
                    .astype(bool)
                )


                inlier_count = int(
                    inlier_mask.sum()
                )


                min_inliers = max(

                    5,

                    int(
                        np.ceil(
                            valid_kp_count
                            *
                            MIN_INLIER_RATIO
                        )
                    )
                )


                if (
                    inlier_count
                    >= min_inliers
                ):

                    try:

                        # =====================================
                        # Reprojection error in image pixels
                        # =====================================

                        H_inverse = np.linalg.inv(
                            H_candidate
                        )


                        pitch_inliers = (
                            target[
                                inlier_mask
                            ]
                            .reshape(
                                -1,
                                1,
                                2
                            )
                        )


                        image_inliers = (
                            source[
                                inlier_mask
                            ]
                        )


                        projected = (
                            cv2.perspectiveTransform(
                                pitch_inliers,
                                H_inverse
                            )
                            .reshape(
                                -1,
                                2
                            )
                        )


                        errors = np.linalg.norm(

                            projected
                            -
                            image_inliers,

                            axis=1
                        )


                        mean_error = float(
                            np.mean(
                                errors
                            )
                        )


                        median_error = float(
                            np.median(
                                errors
                            )
                        )


                        # =====================================
                        # Accept
                        # =====================================

                        if (

                            np.isfinite(
                                mean_error
                            )

                            and

                            np.isfinite(
                                median_error
                            )

                            and

                            mean_error
                            <= MAX_MEAN_REPROJECTION_ERROR

                            and

                            median_error
                            <= MAX_MEDIAN_REPROJECTION_ERROR

                        ):

                            H = (
                                H_candidate
                            )

                            last_H = (
                                H_candidate
                            )

                            fallback_streak = 0


                    except (
                        cv2.error,
                        np.linalg.LinAlgError
                    ):

                        pass


    # ========================================================
    # Short fallback only
    # ========================================================

    if H is None:

        if (
            last_H is not None
            and
            fallback_streak
            < MAX_FALLBACK_FRAMES
        ):

            H = last_H

            fallback_streak += 1

            used_fallback = True

        else:

            H = None

            fallback_streak += 1


    return (
        H,
        last_H,
        fallback_streak,
        used_fallback,
        valid_kp_count,
        inlier_count,
        mean_error,
        median_error
    )


# ============================================================
# Helper:
# Detect Ball
# ============================================================

def detect_ball(frame):

    result = ball_model(

        frame,

        imgsz=BALL_IMGSZ,

        conf=BALL_CONF,

        device=DEVICE,

        verbose=False

    )[0]


    boxes = result.boxes


    if (
        boxes is None
        or
        len(boxes) == 0
    ):

        return None


    xyxy = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )


    confidence = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )


    # ========================================================
    # V1:
    # simply choose highest-confidence candidate
    #
    # 後面如果 false positive 太多，再加入 temporal gating
    # ========================================================

    best_index = int(
        np.argmax(
            confidence
        )
    )


    box = xyxy[
        best_index
    ]


    conf = float(
        confidence[
            best_index
        ]
    )


    x1, y1, x2, y2 = box


    # Ball 使用 bounding-box center
    #
    # 注意：
    # 球若在空中，Homography 投影只是 ground-plane approximation
    image_x = float(
        (x1 + x2) / 2.0
    )

    image_y = float(
        (y1 + y2) / 2.0
    )


    return {

        "box":
            box,

        "confidence":
            conf,

        "image_x":
            image_x,

        "image_y":
            image_y
    }


# ============================================================
# Video
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open: {VIDEO_PATH}"
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
        f"Cannot create: {OUTPUT_VIDEO}"
    )


# ============================================================
# State
# ============================================================

last_H = None

fallback_streak = 0


# Image-space trail
image_trail = deque(
    maxlen=30
)


# Pitch-space trail
pitch_trail = deque(
    maxlen=30
)


rows = []


frame_idx = 0

detected_frames = 0

projected_frames = 0

fallback_frames = 0


print("")
print("========================================")
print("BALL TRACKING V1")
print("========================================")

print("")
print(
    "Frames:",
    total_frames
)

print(
    "FPS:",
    fps
)


# ============================================================
# Main Loop
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:
        break


    annotated = (
        frame.copy()
    )


    # ========================================================
    # 1. Homography
    # ========================================================

    (
        H,
        last_H,
        fallback_streak,
        used_fallback,
        valid_kp_count,
        inlier_count,
        mean_error,
        median_error

    ) = estimate_homography(

        frame,
        last_H,
        fallback_streak
    )


    if used_fallback:

        fallback_frames += 1


    # ========================================================
    # 2. Ball Detection
    # ========================================================

    ball = detect_ball(
        frame
    )


    detected = (
        ball is not None
    )


    confidence = None

    image_x = None
    image_y = None

    pitch_x_cm = None
    pitch_y_cm = None

    pitch_x_m = None
    pitch_y_m = None

    projected = False


    # ========================================================
    # Ball exists
    # ========================================================

    if ball is not None:

        detected_frames += 1


        confidence = (
            ball["confidence"]
        )

        image_x = (
            ball["image_x"]
        )

        image_y = (
            ball["image_y"]
        )


        x1, y1, x2, y2 = (
            ball["box"]
        )


        # ====================================================
        # Draw box
        # ====================================================

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

            (
                0,
                255,
                255
            ),

            2
        )


        cv2.circle(

            annotated,

            (
                int(image_x),
                int(image_y)
            ),

            6,

            (
                0,
                255,
                255
            ),

            -1
        )


        cv2.putText(

            annotated,

            f"Ball {confidence:.2f}",

            (
                int(x1),
                max(
                    int(y1) - 8,
                    20
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (
                0,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


        # ====================================================
        # Image trail
        # ====================================================

        image_trail.append(
            (
                frame_idx,
                image_x,
                image_y
            )
        )


        trail_list = list(
            image_trail
        )


        for i in range(
            1,
            len(trail_list)
        ):

            f1, x_prev, y_prev = (
                trail_list[i - 1]
            )

            f2, x_now, y_now = (
                trail_list[i]
            )


            # 不跨太大的 detection gap 直接連線
            if (
                f2 - f1
                <= 3
            ):

                cv2.line(

                    annotated,

                    (
                        int(x_prev),
                        int(y_prev)
                    ),

                    (
                        int(x_now),
                        int(y_now)
                    ),

                    (
                        0,
                        255,
                        255
                    ),

                    2
                )


        # ====================================================
        # 3. Ball -> Pitch Coordinate
        # ====================================================

        if H is not None:

            image_point = np.array(
                [
                    [
                        [
                            image_x,
                            image_y
                        ]
                    ]
                ],
                dtype=np.float32
            )


            try:

                pitch_point = (
                    cv2.perspectiveTransform(
                        image_point,
                        H
                    )[0][0]
                )


                if np.all(
                    np.isfinite(
                        pitch_point
                    )
                ):

                    px = float(
                        pitch_point[0]
                    )

                    py = float(
                        pitch_point[1]
                    )


                    inside_pitch = (

                        PITCH_X_MIN
                        <= px
                        <= PITCH_X_MAX

                        and

                        PITCH_Y_MIN
                        <= py
                        <= PITCH_Y_MAX
                    )


                    if inside_pitch:

                        projected = True

                        projected_frames += 1


                        pitch_x_cm = px

                        pitch_y_cm = py

                        pitch_x_m = (
                            px / 100.0
                        )

                        pitch_y_m = (
                            py / 100.0
                        )


                        pitch_trail.append(
                            (
                                frame_idx,
                                px,
                                py
                            )
                        )


            except cv2.error:

                pass


    # ========================================================
    # 4. Radar
    # ========================================================

    if projected:

        radar = draw_pitch(
            config=CONFIG
        )


        # Current ball
        current_xy = np.array(
            [
                [
                    pitch_x_cm,
                    pitch_y_cm
                ]
            ],
            dtype=np.float32
        )


        radar = draw_points_on_pitch(

            config=CONFIG,

            xy=current_xy,

            face_color=sv.Color.from_hex(
                "#FFD700"
            ),

            radius=20,

            pitch=radar
        )


        # ====================================================
        # Trail on radar
        # ====================================================

        if len(pitch_trail) > 1:

            trail = list(
                pitch_trail
            )


            # draw circles manually on radar coordinates
            #
            # 我們先只顯示當前點；
            # CSV 才是後面真正做 trajectory smoothing 的來源。


        radar = sv.resize_image(

            radar,

            (
                width // 3,
                height // 3
            )
        )


        radar_h, radar_w = (
            radar.shape[:2]
        )


        rect = sv.Rect(

            x=width - radar_w - 20,

            y=height - radar_h - 20,

            width=radar_w,

            height=radar_h
        )


        annotated = sv.draw_image(

            annotated,

            radar,

            opacity=0.80,

            rect=rect
        )


    # ========================================================
    # Status Text
    # ========================================================

    if detected:

        status = (
            f"Ball conf={confidence:.2f}"
        )

    else:

        status = (
            "Ball: not detected"
        )


    cv2.putText(

        annotated,

        status,

        (
            20,
            35
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (
            0,
            255,
            255
        ),

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # CSV
    # ========================================================

    rows.append(
        {
            "frame":
                frame_idx,

            "time_sec":
                frame_idx / fps,

            "detected":
                detected,

            "confidence":
                confidence,

            "image_x":
                image_x,

            "image_y":
                image_y,

            "projected":
                projected,

            "pitch_x_cm":
                pitch_x_cm,

            "pitch_y_cm":
                pitch_y_cm,

            "pitch_x_m":
                pitch_x_m,

            "pitch_y_m":
                pitch_y_m,

            "valid_keypoints":
                valid_kp_count,

            "inliers":
                inlier_count,

            "mean_reprojection_error_px":
                mean_error,

            "median_reprojection_error_px":
                median_error,

            "homography_fallback":
                used_fallback
        }
    )


    # ========================================================
    # Debug
    # ========================================================

    if frame_idx % 30 == 0:

        conf_text = (

            "N/A"

            if confidence is None

            else

            f"{confidence:.2f}"
        )


        error_text = (

            "N/A"

            if mean_error is None

            else

            f"{mean_error:.2f}px"
        )


        print(

            f"frame={frame_idx:04d} | "

            f"ball={detected} | "

            f"conf={conf_text} | "

            f"projected={projected} | "

            f"kp={valid_kp_count:02d} | "

            f"inliers={inlier_count:02d} | "

            f"err={error_text} | "

            f"fallback={used_fallback}"
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

    "detected",
    "confidence",

    "image_x",
    "image_y",

    "projected",

    "pitch_x_cm",
    "pitch_y_cm",

    "pitch_x_m",
    "pitch_y_m",

    "valid_keypoints",
    "inliers",

    "mean_reprojection_error_px",
    "median_reprojection_error_px",

    "homography_fallback"
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer_csv = csv.DictWriter(

        f,

        fieldnames=columns
    )

    writer_csv.writeheader()

    writer_csv.writerows(
        rows
    )


# ============================================================
# Statistics
# ============================================================

detection_rate = (

    detected_frames
    /
    frame_idx
    *
    100.0

    if frame_idx > 0

    else 0.0
)


projection_rate = (

    projected_frames
    /
    frame_idx
    *
    100.0

    if frame_idx > 0

    else 0.0
)


print("")
print("========================================")
print("BALL TRACKING SUMMARY")
print("========================================")

print("")
print(
    "Frames:",
    frame_idx
)

print(
    "Ball detected:",
    detected_frames
)

print(
    "Detection rate:",
    f"{detection_rate:.1f}%"
)

print(
    "Ball projected:",
    projected_frames
)

print(
    "Projection rate:",
    f"{projection_rate:.1f}%"
)

print(
    "Homography fallback frames:",
    fallback_frames
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
