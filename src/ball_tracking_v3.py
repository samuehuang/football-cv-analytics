import os
import csv
from collections import deque

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO

from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
)

from sports.configs.soccer import (
    SoccerPitchConfiguration,
)


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

BALL_MODEL_PATH = "models/football-ball-detection.pt"
PITCH_MODEL_PATH = "models/football-pitch-detection.pt"

OUTPUT_VIDEO = "outputs/ball_tracking_v3.mp4"
OUTPUT_CSV = "outputs/ball_tracking_v3.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Device
# ============================================================

DEVICE = "mps"


# ============================================================
# Models
# ============================================================

ball_model = YOLO(BALL_MODEL_PATH)
pitch_model = YOLO(PITCH_MODEL_PATH)

CONFIG = SoccerPitchConfiguration()


# ============================================================
# Full-frame Ball Detection
# ============================================================

BALL_CONF = 0.10
BALL_IMGSZ = 1280


# ============================================================
# Adaptive Tiled Fallback
# ============================================================

# 3 columns x 2 rows
TILE_COLS = 3
TILE_ROWS = 2

# overlap between neighboring tiles
TILE_OVERLAP = 0.20

# crop inference resolution
TILE_IMGSZ = 960

# tiled inference threshold
TILE_CONF = 0.08

# overlapping tiles may detect the same ball multiple times
TILE_DEDUP_CENTER_PX = 20.0


# ============================================================
# Temporal Association
# ============================================================

# Very loose upper gate.
# Used to remove obvious false teleports.
MAX_BALL_SPEED = 60.0

# 5 frames @ 25 fps ≈ 0.20 sec
MAX_ASSOCIATION_GAP = 5

# Minimum confidence for initial acquisition
# or reacquisition after a long gap
REACQUIRE_CONF = 0.45


# Candidate score weights
MOTION_WEIGHT = 0.55
LAST_POSITION_WEIGHT = 0.25
CONFIDENCE_WEIGHT = 0.20


# ============================================================
# Homography
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
    dtype=np.float32,
)


PITCH_X_MIN = float(
    np.min(pitch_vertices[:, 0])
)

PITCH_X_MAX = float(
    np.max(pitch_vertices[:, 0])
)

PITCH_Y_MIN = float(
    np.min(pitch_vertices[:, 1])
)

PITCH_Y_MAX = float(
    np.max(pitch_vertices[:, 1])
)


# ============================================================
# Homography Estimation
# ============================================================

def estimate_homography(
    frame,
    last_H,
    fallback_streak,
):

    result = pitch_model(
        frame,
        device=DEVICE,
        verbose=False,
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

            conf = keypoints.confidence[0]

        else:

            conf = np.ones(
                len(pts),
                dtype=np.float32,
            )


        valid = (
            (pts[:, 0] > 1)
            &
            (pts[:, 1] > 1)
            &
            (conf >= KP_CONF_THRESHOLD)
        )


        source = (
            pts[valid]
            .astype(np.float32)
        )


        target = (
            pitch_vertices[valid]
            .astype(np.float32)
        )


        valid_kp_count = len(source)


        # ====================================================
        # RANSAC Homography
        # ====================================================

        if valid_kp_count >= MIN_KEYPOINTS:

            H_candidate, inliers = (
                cv2.findHomography(
                    source,
                    target,
                    cv2.RANSAC,
                    RANSAC_THRESHOLD,
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
                    ),
                )


                if inlier_count >= min_inliers:

                    try:

                        # image -> pitch H
                        # invert it to calculate error
                        # back in image pixels
                        H_inverse = np.linalg.inv(
                            H_candidate
                        )


                        target_inliers = (
                            target[inlier_mask]
                            .reshape(-1, 1, 2)
                        )


                        source_inliers = (
                            source[inlier_mask]
                        )


                        projected_image = (
                            cv2.perspectiveTransform(
                                target_inliers,
                                H_inverse,
                            )
                            .reshape(-1, 2)
                        )


                        errors = np.linalg.norm(
                            projected_image
                            -
                            source_inliers,
                            axis=1,
                        )


                        mean_error = float(
                            np.mean(errors)
                        )


                        median_error = float(
                            np.median(errors)
                        )


                        if (
                            np.isfinite(mean_error)
                            and
                            np.isfinite(median_error)
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
                        np.linalg.LinAlgError,
                    ):

                        pass


    # ========================================================
    # Short Homography fallback
    # ========================================================

    if H is None:

        if (
            last_H is not None
            and
            fallback_streak < MAX_FALLBACK_FRAMES
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
        median_error,
    )


# ============================================================
# Convert image-space detection to candidate
# ============================================================

def make_candidate(
    box,
    confidence,
    H,
    source_name,
):

    x1, y1, x2, y2 = box


    image_x = float(
        (x1 + x2) / 2.0
    )

    image_y = float(
        (y1 + y2) / 2.0
    )


    candidate = {
        "box": np.asarray(
            box,
            dtype=np.float32,
        ),

        "confidence": float(
            confidence
        ),

        "image_x": image_x,
        "image_y": image_y,

        "source": source_name,

        "projected": False,

        "pitch_x_cm": None,
        "pitch_y_cm": None,

        "pitch_x_m": None,
        "pitch_y_m": None,
    }


    if H is None:

        return candidate


    image_point = np.array(
        [
            [
                [
                    image_x,
                    image_y,
                ]
            ]
        ],
        dtype=np.float32,
    )


    try:

        pitch_point = (
            cv2.perspectiveTransform(
                image_point,
                H,
            )[0][0]
        )


        if not np.all(
            np.isfinite(
                pitch_point
            )
        ):

            return candidate


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


        if not inside_pitch:

            return candidate


        candidate["projected"] = True

        candidate["pitch_x_cm"] = px
        candidate["pitch_y_cm"] = py

        candidate["pitch_x_m"] = (
            px / 100.0
        )

        candidate["pitch_y_m"] = (
            py / 100.0
        )


    except cv2.error:

        pass


    return candidate


# ============================================================
# Full-frame Ball Detection
# ============================================================

def detect_full_frame(
    frame,
    H,
):

    result = ball_model(
        frame,
        imgsz=BALL_IMGSZ,
        conf=BALL_CONF,
        device=DEVICE,
        verbose=False,
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
        confidences,
    ):

        candidate = make_candidate(
            box=box,
            confidence=confidence,
            H=H,
            source_name="full",
        )


        candidates.append(
            candidate
        )


    return candidates


# ============================================================
# Generate 3x2 Overlapping Tiles
# ============================================================

def generate_tiles(
    frame_width,
    frame_height,
):

    base_w = int(
        np.ceil(
            frame_width / TILE_COLS
        )
    )


    base_h = int(
        np.ceil(
            frame_height / TILE_ROWS
        )
    )


    overlap_x = int(
        base_w
        *
        TILE_OVERLAP
    )


    overlap_y = int(
        base_h
        *
        TILE_OVERLAP
    )


    tiles = []


    for row in range(TILE_ROWS):

        for col in range(TILE_COLS):

            x1 = (
                col * base_w
                -
                overlap_x
            )

            y1 = (
                row * base_h
                -
                overlap_y
            )


            x2 = (
                (col + 1) * base_w
                +
                overlap_x
            )

            y2 = (
                (row + 1) * base_h
                +
                overlap_y
            )


            x1 = max(
                0,
                x1,
            )

            y1 = max(
                0,
                y1,
            )

            x2 = min(
                frame_width,
                x2,
            )

            y2 = min(
                frame_height,
                y2,
            )


            tiles.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )


    return tiles


# ============================================================
# Candidate De-duplication
# ============================================================

def deduplicate_candidates(
    candidates,
):

    if len(candidates) <= 1:

        return candidates


    # highest-confidence candidate first
    candidates = sorted(
        candidates,
        key=lambda c: c["confidence"],
        reverse=True,
    )


    kept = []


    for candidate in candidates:

        candidate_center = np.array(
            [
                candidate["image_x"],
                candidate["image_y"],
            ],
            dtype=np.float32,
        )


        duplicate = False


        for existing in kept:

            existing_center = np.array(
                [
                    existing["image_x"],
                    existing["image_y"],
                ],
                dtype=np.float32,
            )


            distance_px = float(
                np.linalg.norm(
                    candidate_center
                    -
                    existing_center
                )
            )


            if (
                distance_px
                <= TILE_DEDUP_CENTER_PX
            ):

                duplicate = True

                break


        if not duplicate:

            kept.append(
                candidate
            )


    return kept


# ============================================================
# Tiled Ball Detection
# ============================================================

def detect_tiled(
    frame,
    H,
):

    frame_height, frame_width = (
        frame.shape[:2]
    )


    tiles = generate_tiles(
        frame_width,
        frame_height,
    )


    candidates = []


    for tile_index, (
        tx1,
        ty1,
        tx2,
        ty2,

    ) in enumerate(tiles):


        crop = frame[
            ty1:ty2,
            tx1:tx2,
        ]


        if crop.size == 0:

            continue


        result = ball_model(
            crop,
            imgsz=TILE_IMGSZ,
            conf=TILE_CONF,
            device=DEVICE,
            verbose=False,
        )[0]


        boxes = result.boxes


        if (
            boxes is None
            or
            len(boxes) == 0
        ):

            continue


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


        for local_box, confidence in zip(
            xyxy,
            confidences,
        ):

            x1, y1, x2, y2 = (
                local_box
            )


            # local crop coordinate
            # -> full-frame coordinate
            full_box = np.array(
                [
                    x1 + tx1,
                    y1 + ty1,
                    x2 + tx1,
                    y2 + ty1,
                ],
                dtype=np.float32,
            )


            candidate = make_candidate(
                box=full_box,
                confidence=confidence,
                H=H,
                source_name=(
                    f"tile_{tile_index}"
                ),
            )


            candidates.append(
                candidate
            )


    return deduplicate_candidates(
        candidates
    )


# ============================================================
# Temporal Ball Association
# ============================================================

def associate_ball(
    candidates,
    frame_idx,
    fps,
    history,
):

    projected_candidates = [
        candidate
        for candidate in candidates
        if candidate["projected"]
    ]


    # ========================================================
    # No projected candidate
    # ========================================================

    if len(projected_candidates) == 0:

        return (
            None,
            "no_projected_candidate",
            None,
            None,
            None,
            False,
        )


    # ========================================================
    # No previous ball
    # -> initial acquisition
    # ========================================================

    if len(history) == 0:

        strong_candidates = [
            candidate
            for candidate
            in projected_candidates
            if (
                candidate["confidence"]
                >= REACQUIRE_CONF
            )
        ]


        if len(strong_candidates) == 0:

            return (
                None,
                "waiting_for_reacquire",
                None,
                None,
                None,
                False,
            )


        best = max(
            strong_candidates,
            key=lambda c: c["confidence"],
        )


        return (
            best,
            "init",
            0.0,
            None,
            None,
            True,
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
    # Gap too long
    # -> new segment / reacquisition
    # ========================================================

    if (
        frame_gap
        > MAX_ASSOCIATION_GAP
    ):

        strong_candidates = [
            candidate
            for candidate
            in projected_candidates
            if (
                candidate["confidence"]
                >= REACQUIRE_CONF
            )
        ]


        if len(strong_candidates) == 0:

            return (
                None,
                "waiting_for_reacquire",
                None,
                None,
                None,
                False,
            )


        best = max(
            strong_candidates,
            key=lambda c: c["confidence"],
        )


        return (
            best,
            "reacquire",
            0.0,
            None,
            None,
            True,
        )


    # ========================================================
    # Short-gap association
    # ========================================================

    dt = (
        frame_gap
        /
        fps
    )


    if dt <= 0:

        return (
            None,
            "invalid_dt",
            None,
            None,
            None,
            False,
        )


    last_position = np.array(
        [
            last["pitch_x_m"],
            last["pitch_y_m"],
        ],
        dtype=np.float32,
    )


    # Default:
    # assume stationary if only one previous observation
    predicted_position = (
        last_position.copy()
    )


    # ========================================================
    # Constant-velocity prediction
    # ========================================================

    if len(history) >= 2:

        previous = history[-2]


        previous_gap = (
            last["frame"]
            -
            previous["frame"]
        )


        previous_dt = (
            previous_gap
            /
            fps
        )


        if (
            previous_dt > 0
            and
            previous_gap
            <= MAX_ASSOCIATION_GAP
        ):

            previous_position = np.array(
                [
                    previous["pitch_x_m"],
                    previous["pitch_y_m"],
                ],
                dtype=np.float32,
            )


            velocity = (
                last_position
                -
                previous_position
            ) / previous_dt


            velocity_norm = float(
                np.linalg.norm(
                    velocity
                )
            )


            # only trust previous velocity
            # when it is itself reasonable
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

    max_allowed_distance = (
        MAX_BALL_SPEED
        *
        dt
    )


    normalization_distance = max(
        max_allowed_distance,
        1.0,
    )


    valid_options = []


    for candidate in projected_candidates:

        candidate_position = np.array(
            [
                candidate["pitch_x_m"],
                candidate["pitch_y_m"],
            ],
            dtype=np.float32,
        )


        distance_from_last = float(
            np.linalg.norm(
                candidate_position
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
        # Hard motion gate
        # ====================================================

        if (
            estimated_speed
            > MAX_BALL_SPEED
        ):

            continue


        prediction_error = float(
            np.linalg.norm(
                candidate_position
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


        valid_options.append(
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
                    ),
            }
        )


    # ========================================================
    # Everything rejected by motion gate
    # ========================================================

    if len(valid_options) == 0:

        return (
            None,
            "rejected_motion",
            None,
            None,
            None,
            False,
        )


    # ========================================================
    # Choose best candidate
    # ========================================================

    best = min(
        valid_options,
        key=lambda option:
            option["score"],
    )


    return (
        best["candidate"],
        "tracked",
        best["score"],
        best["speed"],
        best["prediction_error"],
        False,
    )


# ============================================================
# Video Setup
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
        height,
    ),
)


if not writer.isOpened():

    raise RuntimeError(
        f"Cannot create video: {OUTPUT_VIDEO}"
    )


# ============================================================
# State
# ============================================================

last_H = None
fallback_streak = 0


# only accepted ball observations
history = deque(
    maxlen=2
)


# visual trail
image_trail = deque(
    maxlen=30
)


segment_id = -1


rows = []


frame_idx = 0


# ============================================================
# Statistics
# ============================================================

full_detection_frames = 0

tile_fallback_frames = 0
tile_candidate_frames = 0
tile_rescued_frames = 0

accepted_frames = 0
accepted_full_frames = 0
accepted_tile_frames = 0

rejected_motion_frames = 0

reacquire_count = 0

homography_fallback_frames = 0


print("")
print("========================================")
print("BALL TRACKING V3")
print("Adaptive Tiled Inference")
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
        used_homography_fallback,
        valid_kp_count,
        inlier_count,
        mean_error,
        median_error,

    ) = estimate_homography(
        frame,
        last_H,
        fallback_streak,
    )


    if used_homography_fallback:

        homography_fallback_frames += 1


    # ========================================================
    # 2. Full-frame Ball Candidates
    # ========================================================

    full_candidates = detect_full_frame(
        frame,
        H,
    )


    full_candidate_count = len(
        full_candidates
    )


    full_projected_count = sum(
        1
        for candidate
        in full_candidates
        if candidate["projected"]
    )


    if full_candidate_count > 0:

        full_detection_frames += 1


    # ========================================================
    # 3. First association attempt:
    #    FULL frame only
    # ========================================================

    (
        ball,
        association_mode,
        association_score,
        estimated_speed,
        prediction_error,
        new_segment,

    ) = associate_ball(
        full_candidates,
        frame_idx,
        fps,
        history,
    )


    # preserve the reason why full inference failed
    full_association_mode = (
        association_mode
    )


    # ========================================================
    # Tiled fallback state
    # ========================================================

    tile_fallback_used = False

    tile_candidates = []

    tile_candidate_count = 0
    tile_projected_count = 0

    accepted_from_tile = False


    # ========================================================
    # 4. Adaptive Tiled Fallback
    #
    # ONLY run when full-frame association failed.
    # ========================================================

    if ball is None:

        tile_fallback_used = True

        tile_fallback_frames += 1


        tile_candidates = detect_tiled(
            frame,
            H,
        )


        tile_candidate_count = len(
            tile_candidates
        )


        tile_projected_count = sum(
            1
            for candidate
            in tile_candidates
            if candidate["projected"]
        )


        if tile_candidate_count > 0:

            tile_candidate_frames += 1


        # ====================================================
        # IMPORTANT:
        #
        # Merge FULL + TILE candidates.
        #
        # Do not throw away potentially useful full-frame
        # candidates just because the first association
        # attempt failed.
        # ====================================================

        merged_candidates = (
            full_candidates
            +
            tile_candidates
        )


        merged_candidates = (
            deduplicate_candidates(
                merged_candidates
            )
        )


        (
            fallback_ball,
            fallback_mode,
            fallback_score,
            fallback_speed,
            fallback_prediction_error,
            fallback_new_segment,

        ) = associate_ball(
            merged_candidates,
            frame_idx,
            fps,
            history,
        )


        if fallback_ball is not None:

            ball = fallback_ball

            association_mode = (
                f"{fallback_mode}_fallback"
            )

            association_score = (
                fallback_score
            )

            estimated_speed = (
                fallback_speed
            )

            prediction_error = (
                fallback_prediction_error
            )

            new_segment = (
                fallback_new_segment
            )


            accepted_from_tile = (
                ball["source"]
                != "full"
            )


            if accepted_from_tile:

                tile_rescued_frames += 1


        else:

            # final failure reason
            association_mode = (
                f"{fallback_mode}_fallback"
            )

            association_score = None
            estimated_speed = None
            prediction_error = None
            new_segment = False


    # ========================================================
    # 5. Draw Detector Candidates
    # ========================================================

    # FULL candidates = gray
    for candidate in full_candidates:

        x1, y1, x2, y2 = (
            candidate["box"]
        )


        cv2.rectangle(
            annotated,
            (
                int(x1),
                int(y1),
            ),
            (
                int(x2),
                int(y2),
            ),
            (
                150,
                150,
                150,
            ),
            1,
        )


    # TILE candidates = cyan
    for candidate in tile_candidates:

        x1, y1, x2, y2 = (
            candidate["box"]
        )


        cv2.rectangle(
            annotated,
            (
                int(x1),
                int(y1),
            ),
            (
                int(x2),
                int(y2),
            ),
            (
                255,
                200,
                0,
            ),
            1,
        )


    # ========================================================
    # 6. Accepted Ball
    # ========================================================

    accepted = (
        ball is not None
    )


    if (
        accepted
        and
        new_segment
    ):

        segment_id += 1

        history.clear()

        image_trail.clear()


        if (
            "reacquire"
            in association_mode
        ):

            reacquire_count += 1


    confidence = None

    image_x = None
    image_y = None

    pitch_x_cm = None
    pitch_y_cm = None

    pitch_x_m = None
    pitch_y_m = None

    ball_source = None


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


        ball_source = (
            ball["source"]
        )


        if ball_source == "full":

            accepted_full_frames += 1

        else:

            accepted_tile_frames += 1


        # ====================================================
        # Update association history
        # ====================================================

        history.append(
            {
                "frame":
                    frame_idx,

                "pitch_x_m":
                    pitch_x_m,

                "pitch_y_m":
                    pitch_y_m,
            }
        )


        # ====================================================
        # Update visual trail
        # ====================================================

        image_trail.append(
            (
                frame_idx,
                image_x,
                image_y,
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
                int(y1),
            ),
            (
                int(x2),
                int(y2),
            ),
            (
                0,
                255,
                255,
            ),
            3,
        )


        cv2.circle(
            annotated,
            (
                int(image_x),
                int(image_y),
            ),
            7,
            (
                0,
                255,
                255,
            ),
            -1,
        )


        source_label = (
            "FULL"
            if ball_source == "full"
            else
            "TILE"
        )


        cv2.putText(
            annotated,
            (
                f"BALL {confidence:.2f} "
                f"[{source_label}]"
            ),
            (
                int(x1),
                max(
                    int(y1) - 10,
                    20,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )


    else:

        if (
            "rejected_motion"
            in association_mode
        ):

            rejected_motion_frames += 1


    # ========================================================
    # 7. Image-space Trail
    # ========================================================

    trail = list(
        image_trail
    )


    for i in range(
        1,
        len(trail),
    ):

        f1, x1, y1 = (
            trail[i - 1]
        )


        f2, x2, y2 = (
            trail[i]
        )


        if (
            f2 - f1
            <= MAX_ASSOCIATION_GAP
        ):

            cv2.line(
                annotated,
                (
                    int(x1),
                    int(y1),
                ),
                (
                    int(x2),
                    int(y2),
                ),
                (
                    0,
                    255,
                    255,
                ),
                2,
            )


    # ========================================================
    # 8. Radar
    # ========================================================

    if accepted:

        radar = draw_pitch(
            config=CONFIG
        )


        ball_xy = np.array(
            [
                [
                    pitch_x_cm,
                    pitch_y_cm,
                ]
            ],
            dtype=np.float32,
        )


        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=ball_xy,
            face_color=sv.Color.from_hex(
                "#FFD700"
            ),
            radius=20,
            pitch=radar,
        )


        radar = sv.resize_image(
            radar,
            (
                width // 3,
                height // 3,
            ),
        )


        radar_h, radar_w = (
            radar.shape[:2]
        )


        rect = sv.Rect(
            x=width - radar_w - 20,
            y=height - radar_h - 20,
            width=radar_w,
            height=radar_h,
        )


        annotated = sv.draw_image(
            annotated,
            radar,
            opacity=0.80,
            rect=rect,
        )


    # ========================================================
    # 9. Status Overlay
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
            255,
        )


    else:

        status = (
            f"Ball: missing | "
            f"{association_mode}"
        )


        status_color = (
            0,
            120,
            255,
        )


    cv2.putText(
        annotated,
        status,
        (
            20,
            35,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA,
    )


    cv2.putText(
        annotated,
        (
            f"Full={full_candidate_count} "
            f"| Tile={tile_candidate_count} "
            f"| Fallback={tile_fallback_used}"
        ),
        (
            20,
            65,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )


    # ========================================================
    # 10. CSV Row
    # ========================================================

    rows.append(
        {
            "frame":
                frame_idx,

            "time_sec":
                frame_idx / fps,

            "full_candidate_count":
                full_candidate_count,

            "full_projected_count":
                full_projected_count,

            "full_association_mode":
                full_association_mode,

            "tile_fallback_used":
                tile_fallback_used,

            "tile_candidate_count":
                tile_candidate_count,

            "tile_projected_count":
                tile_projected_count,

            "accepted_from_tile":
                accepted_from_tile,

            # compatibility with quality analysis
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

            "ball_source":
                ball_source,

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
                used_homography_fallback,
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


        source_text = (
            "-"
            if ball_source is None
            else
            ball_source
        )


        print(
            f"frame={frame_idx:04d} | "
            f"full={full_candidate_count} | "
            f"tile={tile_candidate_count} | "
            f"fallback={tile_fallback_used} | "
            f"accepted={accepted} | "
            f"source={source_text} | "
            f"mode={association_mode} | "
            f"speed={speed_text}"
        )


    writer.write(
        annotated
    )


    frame_idx += 1


# ============================================================
# Finish Video
# ============================================================

cap.release()

writer.release()


# ============================================================
# Save CSV
# ============================================================

columns = [
    "frame",
    "time_sec",

    "full_candidate_count",
    "full_projected_count",
    "full_association_mode",

    "tile_fallback_used",
    "tile_candidate_count",
    "tile_projected_count",

    "accepted_from_tile",

    "detected",
    "confidence",

    "image_x",
    "image_y",

    "projected",

    "pitch_x_cm",
    "pitch_y_cm",

    "pitch_x_m",
    "pitch_y_m",

    "ball_source",

    "association_mode",
    "association_score",

    "estimated_speed_from_last_mps",
    "prediction_error_m",

    "segment_id",

    "valid_keypoints",
    "inliers",

    "mean_reprojection_error_px",
    "median_reprojection_error_px",

    "homography_fallback",
]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    csv_writer = csv.DictWriter(
        f,
        fieldnames=columns,
    )


    csv_writer.writeheader()

    csv_writer.writerows(
        rows
    )


# ============================================================
# Summary
# ============================================================

full_detection_rate = (
    full_detection_frames
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


tile_rescue_rate = (
    tile_rescued_frames
    /
    tile_fallback_frames
    *
    100.0
    if tile_fallback_frames > 0
    else 0.0
)


print("")
print("========================================")
print("BALL TRACKING V3 SUMMARY")
print("========================================")

print("")
print(
    "Frames:",
    frame_idx
)

print("")
print(
    "Full-frame detection frames:",
    full_detection_frames
)

print(
    "Full-frame detection rate:",
    f"{full_detection_rate:.1f}%"
)

print("")
print(
    "Tile fallback invoked:",
    tile_fallback_frames
)

print(
    "Tile fallback found candidates:",
    tile_candidate_frames
)

print(
    "Tile-rescued accepted frames:",
    tile_rescued_frames
)

print(
    "Tile rescue rate:",
    f"{tile_rescue_rate:.1f}%"
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
    "Accepted from full frame:",
    accepted_full_frames
)

print(
    "Accepted from tile:",
    accepted_tile_frames
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
    homography_fallback_frames
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