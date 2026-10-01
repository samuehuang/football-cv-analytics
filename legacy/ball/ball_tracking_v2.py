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

OUTPUT_VIDEO = "outputs/ball_tracking_v2.mp4"
OUTPUT_CSV = "outputs/ball_tracking_v2.csv"

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
# Ball Detector Parameters
# ============================================================

# Detector threshold 保持低，
# 因為 temporal association 會幫我們篩候選
BALL_CONF = 0.10

BALL_IMGSZ = 1280


# ============================================================
# Temporal Association Parameters
# ============================================================

# 只拿來排除明顯不可能的 teleport
#
# 60 m/s = 216 km/h
#
# 已經是非常寬鬆的 upper gate。
MAX_BALL_SPEED = 60.0


# 最多用舊 trajectory association 幾個 frame
#
# 5 frames @25fps ≈ 0.20 sec
MAX_ASSOCIATION_GAP = 5


# 第一次偵測 / 長 gap 後重新取得球時，
# confidence 至少要有這麼高
REACQUIRE_CONF = 0.45


# Candidate scoring
MOTION_WEIGHT = 0.55
LAST_POSITION_WEIGHT = 0.25
CONFIDENCE_WEIGHT = 0.20


# ============================================================
# Homography Parameters
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
# Homography
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


    if len(keypoints.xy) > 0:

        pts = keypoints.xy[0]


        if keypoints.confidence is not None:

            conf = keypoints.confidence[0]

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

                        H_inverse = np.linalg.inv(
                            H_candidate
                        )


                        target_inliers = (
                            target[
                                inlier_mask
                            ]
                            .reshape(
                                -1,
                                1,
                                2
                            )
                        )


                        source_inliers = (
                            source[
                                inlier_mask
                            ]
                        )


                        projected_image = (
                            cv2.perspectiveTransform(
                                target_inliers,
                                H_inverse
                            )
                            .reshape(
                                -1,
                                2
                            )
                        )


                        errors = np.linalg.norm(

                            projected_image
                            -
                            source_inliers,

                            axis=1
                        )


                        mean_error = float(
                            np.mean(errors)
                        )

                        median_error = float(
                            np.median(errors)
                        )


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

                            H = H_candidate

                            last_H = H_candidate

                            fallback_streak = 0


                    except (
                        cv2.error,
                        np.linalg.LinAlgError
                    ):

                        pass


    # ========================================================
    # Short fallback
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
# Ball Candidate Detection
# ============================================================

def detect_ball_candidates(
    frame,
    H
):

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

        return []


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


    candidates = []


    for box, confidence in zip(
        xyxy,
        confidences
    ):

        x1, y1, x2, y2 = box


        image_x = float(
            (x1 + x2) / 2.0
        )

        image_y = float(
            (y1 + y2) / 2.0
        )


        candidate = {

            "box":
                box,

            "confidence":
                float(confidence),

            "image_x":
                image_x,

            "image_y":
                image_y,

            "projected":
                False,

            "pitch_x_cm":
                None,

            "pitch_y_cm":
                None,

            "pitch_x_m":
                None,

            "pitch_y_m":
                None
        }


        # ====================================================
        # Project candidate to pitch
        # ====================================================

        if H is not None:

            point = np.array(
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
                        point,
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

                        candidate[
                            "projected"
                        ] = True


                        candidate[
                            "pitch_x_cm"
                        ] = px

                        candidate[
                            "pitch_y_cm"
                        ] = py

                        candidate[
                            "pitch_x_m"
                        ] = px / 100.0

                        candidate[
                            "pitch_y_m"
                        ] = py / 100.0


            except cv2.error:

                pass


        candidates.append(
            candidate
        )


    return candidates


# ============================================================
# Temporal Association
# ============================================================

def associate_ball(
    candidates,
    frame_idx,
    fps,
    history
):

    projected_candidates = [

        c
        for c in candidates
        if c["projected"]

    ]


    if len(projected_candidates) == 0:

        return (
            None,
            "no_projected_candidate",
            None,
            None,
            None,
            False
        )


    # ========================================================
    # No previous accepted ball
    # -> initialize
    # ========================================================

    if len(history) == 0:

        strong = [

            c
            for c in projected_candidates
            if c["confidence"]
            >= REACQUIRE_CONF
        ]


        if len(strong) == 0:

            return (
                None,
                "waiting_for_reacquire",
                None,
                None,
                None,
                False
            )


        best = max(
            strong,
            key=lambda c: c["confidence"]
        )


        return (
            best,
            "init",
            0.0,
            None,
            None,
            True
        )


    # ========================================================
    # Previous accepted ball
    # ========================================================

    last = history[-1]


    last_frame = int(
        last["frame"]
    )


    frame_gap = (
        frame_idx
        -
        last_frame
    )


    # ========================================================
    # Gap too large
    #
    # Do NOT connect the old trajectory.
    # Start a new segment.
    # ========================================================

    if (
        frame_gap
        > MAX_ASSOCIATION_GAP
    ):

        strong = [

            c
            for c in projected_candidates
            if c["confidence"]
            >= REACQUIRE_CONF
        ]


        if len(strong) == 0:

            return (
                None,
                "waiting_for_reacquire",
                None,
                None,
                None,
                False
            )


        best = max(
            strong,
            key=lambda c: c["confidence"]
        )


        return (
            best,
            "reacquire",
            0.0,
            None,
            None,
            True
        )


    # ========================================================
    # Short gap association
    # ========================================================

    dt = (
        frame_gap
        / fps
    )


    if dt <= 0:

        return (
            None,
            "invalid_dt",
            None,
            None,
            None,
            False
        )


    last_position = np.array(

        [
            last["pitch_x_m"],
            last["pitch_y_m"]
        ],

        dtype=np.float32
    )


    # ========================================================
    # Constant velocity prediction
    # ========================================================

    predicted_position = (
        last_position.copy()
    )


    if len(history) >= 2:

        previous = history[-2]


        history_dt = (

            last["frame"]
            -
            previous["frame"]

        ) / fps


        if (
            history_dt > 0
            and
            (
                last["frame"]
                -
                previous["frame"]
            )
            <= MAX_ASSOCIATION_GAP
        ):

            previous_position = np.array(

                [
                    previous["pitch_x_m"],
                    previous["pitch_y_m"]
                ],

                dtype=np.float32
            )


            velocity = (

                last_position
                -
                previous_position

            ) / history_dt


            velocity_norm = float(
                np.linalg.norm(
                    velocity
                )
            )


            # Previous velocity itself should already be sane,
            # but cap prediction just in case.
            if (
                velocity_norm
                <= MAX_BALL_SPEED
            ):

                predicted_position = (

                    last_position
                    +
                    velocity * dt
                )


    # ========================================================
    # Candidate scoring
    # ========================================================

    accepted_options = []


    max_allowed_distance = (
        MAX_BALL_SPEED
        *
        dt
    )


    normalization_distance = max(
        max_allowed_distance,
        1.0
    )


    for candidate in projected_candidates:

        position = np.array(

            [
                candidate[
                    "pitch_x_m"
                ],

                candidate[
                    "pitch_y_m"
                ]
            ],

            dtype=np.float32
        )


        distance_from_last = float(
            np.linalg.norm(
                position
                -
                last_position
            )
        )


        estimated_speed = (
            distance_from_last
            /
            dt
        )


        # ====================================================
        # Hard physical gate
        # ====================================================

        if (
            estimated_speed
            > MAX_BALL_SPEED
        ):

            continue


        prediction_error = float(
            np.linalg.norm(
                position
                -
                predicted_position
            )
        )


        motion_cost = (
            prediction_error
            /
            normalization_distance
        )


        last_position_cost = (
            distance_from_last
            /
            normalization_distance
        )


        confidence_cost = (
            1.0
            -
            candidate["confidence"]
        )


        score = (

            MOTION_WEIGHT
            *
            motion_cost

            +

            LAST_POSITION_WEIGHT
            *
            last_position_cost

            +

            CONFIDENCE_WEIGHT
            *
            confidence_cost
        )


        accepted_options.append(
            {
                "candidate":
                    candidate,

                "score":
                    float(score),

                "speed":
                    float(
                        estimated_speed
                    ),

                "prediction_error":
                    float(
                        prediction_error
                    )
            }
        )


    # ========================================================
    # Nothing passes temporal gate
    # ========================================================

    if len(
        accepted_options
    ) == 0:

        return (
            None,
            "rejected_motion",
            None,
            None,
            None,
            False
        )


    # ========================================================
    # Best candidate
    # ========================================================

    best = min(
        accepted_options,
        key=lambda x: x["score"]
    )


    return (
        best["candidate"],
        "tracked",
        best["score"],
        best["speed"],
        best["prediction_error"],
        False
    )


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


# Only accepted ball observations
history = deque(
    maxlen=2
)


# For visual trail
image_trail = deque(
    maxlen=30
)

pitch_trail = deque(
    maxlen=30
)


segment_id = -1


rows = []


frame_idx = 0


# ============================================================
# Statistics
# ============================================================

raw_detection_frames = 0
accepted_frames = 0
rejected_motion_frames = 0
reacquire_count = 0

fallback_frames = 0


print("")
print("========================================")
print("BALL TRACKING V2")
print("Temporal Candidate Association")
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
    # 2. Detect ALL ball candidates
    # ========================================================

    candidates = detect_ball_candidates(
        frame,
        H
    )


    raw_candidate_count = len(
        candidates
    )


    projected_candidate_count = sum(

        1
        for c in candidates
        if c["projected"]

    )


    if raw_candidate_count > 0:

        raw_detection_frames += 1


    # ========================================================
    # Draw raw candidates
    #
    # Gray = candidate
    # Yellow = finally accepted ball
    # ========================================================

    for candidate in candidates:

        box = candidate["box"]

        x1, y1, x2, y2 = box


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
                160,
                160,
                160
            ),

            1
        )


        cv2.putText(

            annotated,

            f"{candidate['confidence']:.2f}",

            (
                int(x1),
                max(
                    int(y1) - 3,
                    12
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.35,

            (
                180,
                180,
                180
            ),

            1,

            cv2.LINE_AA
        )


    # ========================================================
    # 3. Temporal association
    # ========================================================

    (
        ball,
        association_mode,
        association_score,
        estimated_speed,
        prediction_error,
        new_segment

    ) = associate_ball(

        candidates,
        frame_idx,
        fps,
        history
    )


    accepted = (
        ball is not None
    )


    # ========================================================
    # Start/restart segment
    # ========================================================

    if (
        accepted
        and
        new_segment
    ):

        segment_id += 1

        history.clear()

        image_trail.clear()

        pitch_trail.clear()


        if association_mode == "reacquire":

            reacquire_count += 1


    # ========================================================
    # Accepted ball
    # ========================================================

    confidence = None

    image_x = None
    image_y = None

    pitch_x_cm = None
    pitch_y_cm = None

    pitch_x_m = None
    pitch_y_m = None


    if accepted:

        accepted_frames += 1


        confidence = (
            ball["confidence"]
        )

        image_x = (
            ball["image_x"]
        )

        image_y = (
            ball["image_y"]
        )

        pitch_x_cm = (
            ball["pitch_x_cm"]
        )

        pitch_y_cm = (
            ball["pitch_y_cm"]
        )

        pitch_x_m = (
            ball["pitch_x_m"]
        )

        pitch_y_m = (
            ball["pitch_y_m"]
        )


        # ====================================================
        # Update history
        # ====================================================

        history.append(
            {
                "frame":
                    frame_idx,

                "pitch_x_m":
                    pitch_x_m,

                "pitch_y_m":
                    pitch_y_m
            }
        )


        image_trail.append(
            (
                frame_idx,
                image_x,
                image_y
            )
        )


        pitch_trail.append(
            (
                frame_idx,
                pitch_x_cm,
                pitch_y_cm
            )
        )


        # ====================================================
        # Draw accepted ball
        # ====================================================

        x1, y1, x2, y2 = (
            ball["box"]
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

            (
                0,
                255,
                255
            ),

            3
        )


        cv2.circle(

            annotated,

            (
                int(image_x),
                int(image_y)
            ),

            7,

            (
                0,
                255,
                255
            ),

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
                    int(y1) - 10,
                    20
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                0,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


    elif association_mode == "rejected_motion":

        rejected_motion_frames += 1


    # ========================================================
    # 4. Image-space trail
    # ========================================================

    trail = list(
        image_trail
    )


    for i in range(
        1,
        len(trail)
    ):

        f1, x1, y1 = (
            trail[i - 1]
        )

        f2, x2, y2 = (
            trail[i]
        )


        # Only connect nearby frames
        if (
            f2 - f1
            <= MAX_ASSOCIATION_GAP
        ):

            cv2.line(

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


    # ========================================================
    # 5. Radar
    # ========================================================

    if accepted:

        radar = draw_pitch(
            config=CONFIG
        )


        ball_xy = np.array(
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

            xy=ball_xy,

            face_color=sv.Color.from_hex(
                "#FFD700"
            ),

            radius=20,

            pitch=radar
        )


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
    # Status
    # ========================================================

    if accepted:

        status = (

            f"Ball: {association_mode} | "
            f"conf={confidence:.2f} | "
            f"segment={segment_id}"
        )

        status_color = (
            0,
            255,
            255
        )

    else:

        status = (

            f"Ball: missing | "
            f"{association_mode}"
        )

        status_color = (
            0,
            120,
            255
        )


    cv2.putText(

        annotated,

        status,

        (
            20,
            35
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        status_color,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        annotated,

        (
            f"Candidates: "
            f"{raw_candidate_count} "
            f"(pitch={projected_candidate_count})"
        ),

        (
            20,
            65
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (
            255,
            255,
            255
        ),

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # CSV
    #
    # detected / projected are kept for compatibility with
    # ball_quality_analysis.py.
    # They now mean ACCEPTED ball observation.
    # ========================================================

    rows.append(
        {
            "frame":
                frame_idx,

            "time_sec":
                frame_idx / fps,

            "raw_candidate_count":
                raw_candidate_count,

            "projected_candidate_count":
                projected_candidate_count,

            "detected":
                accepted,

            "confidence":
                confidence,

            "image_x":
                image_x,

            "image_y":
                image_y,

            "projected":
                accepted,

            "pitch_x_cm":
                pitch_x_cm,

            "pitch_y_cm":
                pitch_y_cm,

            "pitch_x_m":
                pitch_x_m,

            "pitch_y_m":
                pitch_y_m,

            "association_mode":
                association_mode,

            "association_score":
                association_score,

            "estimated_speed_from_last_mps":
                estimated_speed,

            "prediction_error_m":
                prediction_error,

            "segment_id":
                (
                    segment_id
                    if accepted
                    else None
                ),

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

        speed_text = (

            "N/A"

            if estimated_speed is None

            else

            f"{estimated_speed:.1f}"
        )


        print(

            f"frame={frame_idx:04d} | "

            f"raw={raw_candidate_count} | "

            f"pitch={projected_candidate_count} | "

            f"accepted={accepted} | "

            f"mode={association_mode} | "

            f"speed={speed_text} | "

            f"segment={segment_id}"
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

    "raw_candidate_count",
    "projected_candidate_count",

    "detected",
    "confidence",

    "image_x",
    "image_y",

    "projected",

    "pitch_x_cm",
    "pitch_y_cm",

    "pitch_x_m",
    "pitch_y_m",

    "association_mode",
    "association_score",

    "estimated_speed_from_last_mps",
    "prediction_error_m",

    "segment_id",

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

raw_detection_rate = (

    raw_detection_frames
    /
    frame_idx
    *
    100.0

    if frame_idx > 0

    else 0.0
)


accepted_rate = (

    accepted_frames
    /
    frame_idx
    *
    100.0

    if frame_idx > 0

    else 0.0
)


print("")
print("========================================")
print("BALL TRACKING V2 SUMMARY")
print("========================================")

print("")
print(
    "Frames:",
    frame_idx
)

print(
    "Raw detection frames:",
    raw_detection_frames
)

print(
    "Raw detection rate:",
    f"{raw_detection_rate:.1f}%"
)

print("")
print(
    "Accepted ball frames:",
    accepted_frames
)

print(
    "Accepted rate:",
    f"{accepted_rate:.1f}%"
)

print("")
print(
    "Rejected-motion frames:",
    rejected_motion_frames
)

print(
    "Reacquisitions:",
    reacquire_count
)

print(
    "Trajectory segments:",
    segment_id + 1
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
