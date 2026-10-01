import os
import csv
from collections import deque

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

BALL_MODEL_PATH = "models/yolov8m_forzasys_soccer.pt"
PITCH_MODEL_PATH = "models/football-pitch-detection.pt"

OUTPUT_VIDEO = "outputs/ball_tracking_forzasys.mp4"
OUTPUT_CSV = "outputs/ball_tracking_forzasys.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Device / Models
# ============================================================

DEVICE = "mps"

ball_model = YOLO(BALL_MODEL_PATH)
pitch_model = YOLO(PITCH_MODEL_PATH)

CONFIG = SoccerPitchConfiguration()


# ============================================================
# Model Classes
#
# ForzaSys:
# 0 = player
# 1 = ball
# 2 = logo
# ============================================================

BALL_CLASS_ID = 1


# ============================================================
# Ball Detection
# ============================================================

BALL_CONF = 0.10
BALL_IMGSZ = 1280


# ============================================================
# Pitch / Homography
# ============================================================

KP_CONF_THRESHOLD = 0.60

MIN_KEYPOINTS = 6
MIN_INLIER_RATIO = 0.60

RANSAC_THRESHOLD = 250.0

MAX_MEAN_REPROJECTION_ERROR = 30.0
MAX_MEDIAN_REPROJECTION_ERROR = 20.0

MAX_HOMOGRAPHY_FALLBACK = 3


# ============================================================
# Pitch geometry
# ============================================================

pitch_vertices = np.asarray(
    CONFIG.vertices,
    dtype=np.float32
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


# Allow a very small margin around the touchline.
#
# Coordinates are centimeters.
#
# 75 cm is enough to avoid losing detections whose center
# lies almost exactly on the line, while still rejecting
# coaches / benches / spectators.
PITCH_MARGIN_CM = 75.0


# Image-space pitch polygon tolerance.
#
# Slightly outside the line is okay because detections are
# not pixel-perfect.
IMAGE_PITCH_MARGIN_PX = 8.0


# ============================================================
# Temporal tracking
# ============================================================

MAX_ASSOCIATION_GAP = 5


# Very loose physical sanity bound.
# Used only against catastrophic jumps.
MAX_BALL_SPEED_MPS = 65.0


# ============================================================
# Image-space prediction gate
# ============================================================

BASE_IMAGE_GATE_PX = 100.0

IMAGE_GATE_PER_MISSING_FRAME = 65.0


# ============================================================
# Pitch prediction gate
#
# Looser than player tracking because ball may be airborne.
# Homography assumes ground plane.
# ============================================================

BASE_PITCH_GATE_M = 7.0

PITCH_GATE_PER_MISSING_FRAME = 3.0


# ============================================================
# Association scoring
# ============================================================

IMAGE_SCORE_WEIGHT = 0.60
PITCH_SCORE_WEIGHT = 0.25
CONF_SCORE_WEIGHT = 0.15


# ============================================================
# Reacquisition
#
# We do not instantly trust a new object after losing the ball.
# ============================================================

REACQUIRE_MIN_CONF = 0.20

REACQUIRE_CONFIRM_GAP = 2

REACQUIRE_MAX_IMAGE_SPEED_PX_S = 2400.0

REACQUIRE_MAX_PITCH_SPEED_MPS = 65.0


# ============================================================
# Homography estimation
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

    keypoints = sv.KeyPoints.from_ultralytics(
        result
    )

    H = None

    valid_kp_count = 0
    inlier_count = 0

    mean_error = None
    median_error = None

    used_fallback = False


    # --------------------------------------------------------
    # Read keypoints
    # --------------------------------------------------------

    if len(keypoints.xy) > 0:

        points = keypoints.xy[0]


        if keypoints.confidence is not None:

            confidences = (
                keypoints.confidence[0]
            )

        else:

            confidences = np.ones(
                len(points),
                dtype=np.float32
            )


        valid = (

            (points[:, 0] > 1)

            &

            (points[:, 1] > 1)

            &

            (
                confidences
                >= KP_CONF_THRESHOLD
            )
        )


        source = (
            points[valid]
            .astype(np.float32)
        )


        target = (
            pitch_vertices[valid]
            .astype(np.float32)
        )


        valid_kp_count = len(
            source
        )


        # ----------------------------------------------------
        # Homography + RANSAC
        # ----------------------------------------------------

        if valid_kp_count >= MIN_KEYPOINTS:

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
                    np.sum(
                        inlier_mask
                    )
                )


                required_inliers = max(
                    5,
                    int(
                        np.ceil(
                            valid_kp_count
                            *
                            MIN_INLIER_RATIO
                        )
                    )
                )


                if inlier_count >= required_inliers:

                    try:

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


                        projected_back = (
                            cv2.perspectiveTransform(
                                target_inliers,
                                H_inverse
                            )
                            .reshape(-1, 2)
                        )


                        errors = np.linalg.norm(
                            projected_back
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

                            np.isfinite(mean_error)

                            and

                            np.isfinite(median_error)

                            and

                            mean_error
                            <=
                            MAX_MEAN_REPROJECTION_ERROR

                            and

                            median_error
                            <=
                            MAX_MEDIAN_REPROJECTION_ERROR

                        ):

                            H = H_candidate

                            last_H = H_candidate

                            fallback_streak = 0


                    except (
                        np.linalg.LinAlgError,
                        cv2.error
                    ):

                        pass


    # --------------------------------------------------------
    # Short fallback
    # --------------------------------------------------------

    if H is None:

        if (

            last_H is not None

            and

            fallback_streak
            <
            MAX_HOMOGRAPHY_FALLBACK

        ):

            H = last_H

            fallback_streak += 1

            used_fallback = True


        else:

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
# Pitch image polygon
# ============================================================

def get_pitch_image_polygon(H):

    if H is None:

        return None


    try:

        H_inverse = np.linalg.inv(H)

    except np.linalg.LinAlgError:

        return None


    # --------------------------------------------------------
    # Four outer corners of 2D pitch
    # --------------------------------------------------------

    pitch_corners = np.array(
        [
            [
                PITCH_X_MIN,
                PITCH_Y_MIN
            ],

            [
                PITCH_X_MAX,
                PITCH_Y_MIN
            ],

            [
                PITCH_X_MAX,
                PITCH_Y_MAX
            ],

            [
                PITCH_X_MIN,
                PITCH_Y_MAX
            ],
        ],
        dtype=np.float32
    )


    pitch_corners = pitch_corners.reshape(
        -1,
        1,
        2
    )


    try:

        image_corners = (
            cv2.perspectiveTransform(
                pitch_corners,
                H_inverse
            )
            .reshape(-1, 2)
        )

    except cv2.error:

        return None


    if not np.all(
        np.isfinite(image_corners)
    ):

        return None


    return image_corners.astype(
        np.float32
    )


# ============================================================
# Image -> pitch projection
# ============================================================

def project_to_pitch(
    image_x,
    image_y,
    H
):

    if H is None:

        return None


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

        projected = (
            cv2.perspectiveTransform(
                point,
                H
            )[0][0]
        )

    except cv2.error:

        return None


    if not np.all(
        np.isfinite(projected)
    ):

        return None


    pitch_x_cm = float(
        projected[0]
    )

    pitch_y_cm = float(
        projected[1]
    )


    return (
        pitch_x_cm,
        pitch_y_cm
    )


# ============================================================
# Pitch candidate filtering
# ============================================================

def is_inside_pitch(
    image_x,
    image_y,
    pitch_x_cm,
    pitch_y_cm,
    image_polygon
):

    # --------------------------------------------------------
    # 1. Image-space polygon test
    # --------------------------------------------------------

    if image_polygon is None:

        return False


    polygon = image_polygon.reshape(
        -1,
        1,
        2
    )


    signed_distance = cv2.pointPolygonTest(
        polygon,
        (
            float(image_x),
            float(image_y)
        ),
        True
    )


    image_inside = (
        signed_distance
        >=
        -IMAGE_PITCH_MARGIN_PX
    )


    if not image_inside:

        return False


    # --------------------------------------------------------
    # 2. Pitch coordinate bounds
    # --------------------------------------------------------

    pitch_inside = (

        PITCH_X_MIN
        -
        PITCH_MARGIN_CM

        <=

        pitch_x_cm

        <=

        PITCH_X_MAX
        +
        PITCH_MARGIN_CM

        and

        PITCH_Y_MIN
        -
        PITCH_MARGIN_CM

        <=

        pitch_y_cm

        <=

        PITCH_Y_MAX
        +
        PITCH_MARGIN_CM
    )


    return pitch_inside


# ============================================================
# Detect Ball Candidates
# ============================================================

def detect_ball_candidates(
    frame,
    H,
    pitch_polygon
):

    result = ball_model(
        frame,
        imgsz=BALL_IMGSZ,
        conf=BALL_CONF,
        classes=[BALL_CLASS_ID],
        device=DEVICE,
        verbose=False
    )[0]


    boxes = result.boxes


    raw_candidates = []

    valid_candidates = []

    rejected_candidates = []


    if (
        boxes is None
        or
        len(boxes) == 0
    ):

        return (
            raw_candidates,
            valid_candidates,
            rejected_candidates
        )


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


    # ========================================================
    # Candidates
    # ========================================================

    for box, confidence in zip(
        xyxy,
        confidences
    ):

        x1, y1, x2, y2 = box


        image_x = float(
            (
                x1 + x2
            )
            /
            2.0
        )


        image_y = float(
            (
                y1 + y2
            )
            /
            2.0
        )


        candidate = {

            "box":
                np.asarray(
                    box,
                    dtype=np.float32
                ),

            "confidence":
                float(
                    confidence
                ),

            "image_x":
                image_x,

            "image_y":
                image_y,

            "pitch_x_cm":
                None,

            "pitch_y_cm":
                None,

            "pitch_x_m":
                None,

            "pitch_y_m":
                None,

            "pitch_valid":
                False
        }


        raw_candidates.append(
            candidate
        )


        # ----------------------------------------------------
        # Homography missing
        # ----------------------------------------------------

        if H is None:

            rejected_candidates.append(
                candidate
            )

            continue


        projection = project_to_pitch(
            image_x,
            image_y,
            H
        )


        if projection is None:

            rejected_candidates.append(
                candidate
            )

            continue


        (
            pitch_x_cm,
            pitch_y_cm
        ) = projection


        candidate[
            "pitch_x_cm"
        ] = pitch_x_cm


        candidate[
            "pitch_y_cm"
        ] = pitch_y_cm


        candidate[
            "pitch_x_m"
        ] = (
            pitch_x_cm
            /
            100.0
        )


        candidate[
            "pitch_y_m"
        ] = (
            pitch_y_cm
            /
            100.0
        )


        # ----------------------------------------------------
        # Pitch ROI
        # ----------------------------------------------------

        pitch_valid = is_inside_pitch(

            image_x,
            image_y,

            pitch_x_cm,
            pitch_y_cm,

            pitch_polygon
        )


        candidate[
            "pitch_valid"
        ] = pitch_valid


        if pitch_valid:

            valid_candidates.append(
                candidate
            )

        else:

            rejected_candidates.append(
                candidate
            )


    return (
        raw_candidates,
        valid_candidates,
        rejected_candidates
    )


# ============================================================
# Prediction
# ============================================================

def predict_position(
    history,
    frame_idx,
    fps
):

    last = history[-1]


    frame_gap = (
        frame_idx
        -
        int(
            last["frame"]
        )
    )


    dt = (
        frame_gap
        /
        fps
    )


    predicted_image = np.array(
        [
            last["image_x"],
            last["image_y"]
        ],
        dtype=np.float32
    )


    predicted_pitch = np.array(
        [
            last["pitch_x_m"],
            last["pitch_y_m"]
        ],
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Constant velocity prediction
    # --------------------------------------------------------

    if len(history) >= 2:

        previous = history[-2]


        previous_frame_gap = (

            int(
                last["frame"]
            )

            -

            int(
                previous["frame"]
            )
        )


        previous_dt = (
            previous_frame_gap
            /
            fps
        )


        if (

            previous_dt > 0

            and

            previous_frame_gap
            <=
            MAX_ASSOCIATION_GAP

        ):

            previous_image = np.array(
                [
                    previous["image_x"],
                    previous["image_y"]
                ],
                dtype=np.float32
            )


            last_image = np.array(
                [
                    last["image_x"],
                    last["image_y"]
                ],
                dtype=np.float32
            )


            image_velocity = (
                (
                    last_image
                    -
                    previous_image
                )
                /
                previous_dt
            )


            previous_pitch = np.array(
                [
                    previous["pitch_x_m"],
                    previous["pitch_y_m"]
                ],
                dtype=np.float32
            )


            last_pitch = np.array(
                [
                    last["pitch_x_m"],
                    last["pitch_y_m"]
                ],
                dtype=np.float32
            )


            pitch_velocity = (
                (
                    last_pitch
                    -
                    previous_pitch
                )
                /
                previous_dt
            )


            pitch_speed = float(
                np.linalg.norm(
                    pitch_velocity
                )
            )


            # Do not extrapolate from insane velocity.
            if (
                pitch_speed
                <=
                MAX_BALL_SPEED_MPS
            ):

                predicted_image = (
                    last_image
                    +
                    image_velocity
                    *
                    dt
                )


                predicted_pitch = (
                    last_pitch
                    +
                    pitch_velocity
                    *
                    dt
                )


    return (
        frame_gap,
        dt,
        predicted_image,
        predicted_pitch
    )


# ============================================================
# Active temporal association
# ============================================================

def associate_candidate(
    candidates,
    history,
    frame_idx,
    fps
):

    if len(history) == 0:

        return (
            None,
            None,
            None,
            None
        )


    (
        frame_gap,
        dt,
        predicted_image,
        predicted_pitch

    ) = predict_position(
        history,
        frame_idx,
        fps
    )


    if (
        frame_gap <= 0
        or
        frame_gap > MAX_ASSOCIATION_GAP
    ):

        return (
            None,
            None,
            None,
            None
        )


    last = history[-1]


    last_pitch = np.array(
        [
            last["pitch_x_m"],
            last["pitch_y_m"]
        ],
        dtype=np.float32
    )


    image_gate = (

        BASE_IMAGE_GATE_PX

        +

        IMAGE_GATE_PER_MISSING_FRAME
        *
        (
            frame_gap - 1
        )
    )


    pitch_gate = (

        BASE_PITCH_GATE_M

        +

        PITCH_GATE_PER_MISSING_FRAME
        *
        (
            frame_gap - 1
        )
    )


    options = []


    for candidate in candidates:

        image_position = np.array(
            [
                candidate["image_x"],
                candidate["image_y"]
            ],
            dtype=np.float32
        )


        pitch_position = np.array(
            [
                candidate["pitch_x_m"],
                candidate["pitch_y_m"]
            ],
            dtype=np.float32
        )


        image_residual = float(
            np.linalg.norm(
                image_position
                -
                predicted_image
            )
        )


        pitch_residual = float(
            np.linalg.norm(
                pitch_position
                -
                predicted_pitch
            )
        )


        distance_from_last = float(
            np.linalg.norm(
                pitch_position
                -
                last_pitch
            )
        )


        speed_mps = (
            distance_from_last
            /
            dt
        )


        # ----------------------------------------------------
        # Hard gates
        # ----------------------------------------------------

        if (
            speed_mps
            >
            MAX_BALL_SPEED_MPS
        ):

            continue


        if (
            image_residual
            >
            image_gate
        ):

            continue


        if (
            pitch_residual
            >
            pitch_gate
        ):

            continue


        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        image_cost = (
            image_residual
            /
            max(
                image_gate,
                1.0
            )
        )


        pitch_cost = (
            pitch_residual
            /
            max(
                pitch_gate,
                0.5
            )
        )


        confidence_cost = (
            1.0
            -
            candidate["confidence"]
        )


        score = (

            IMAGE_SCORE_WEIGHT
            *
            image_cost

            +

            PITCH_SCORE_WEIGHT
            *
            pitch_cost

            +

            CONF_SCORE_WEIGHT
            *
            confidence_cost
        )


        options.append(
            (
                score,
                candidate,
                speed_mps,
                image_residual,
                pitch_residual
            )
        )


    if not options:

        return (
            None,
            None,
            None,
            None
        )


    options.sort(
        key=lambda x:
            x[0]
    )


    (
        score,
        candidate,
        speed_mps,
        image_residual,
        pitch_residual

    ) = options[0]


    return (
        candidate,
        float(speed_mps),
        float(image_residual),
        float(pitch_residual)
    )


# ============================================================
# Reacquisition seed
# ============================================================

def choose_reacquisition_seed(
    candidates
):

    eligible = [

        candidate

        for candidate
        in candidates

        if (
            candidate["confidence"]
            >=
            REACQUIRE_MIN_CONF
        )
    ]


    if not eligible:

        return None


    eligible.sort(
        key=lambda candidate:
            candidate["confidence"],
        reverse=True
    )


    return eligible[0]


# ============================================================
# Reacquisition confirmation
# ============================================================

def confirm_reacquisition(
    pending,
    candidates,
    frame_idx,
    fps
):

    if pending is None:

        return None


    frame_gap = (

        frame_idx

        -

        pending["frame"]
    )


    if (
        frame_gap <= 0
        or
        frame_gap > REACQUIRE_CONFIRM_GAP
    ):

        return None


    dt = (
        frame_gap
        /
        fps
    )


    pending_image = np.array(
        [
            pending["image_x"],
            pending["image_y"]
        ],
        dtype=np.float32
    )


    pending_pitch = np.array(
        [
            pending["pitch_x_m"],
            pending["pitch_y_m"]
        ],
        dtype=np.float32
    )


    options = []


    for candidate in candidates:

        if (
            candidate["confidence"]
            <
            REACQUIRE_MIN_CONF
        ):

            continue


        image_position = np.array(
            [
                candidate["image_x"],
                candidate["image_y"]
            ],
            dtype=np.float32
        )


        pitch_position = np.array(
            [
                candidate["pitch_x_m"],
                candidate["pitch_y_m"]
            ],
            dtype=np.float32
        )


        image_distance = float(
            np.linalg.norm(
                image_position
                -
                pending_image
            )
        )


        pitch_distance = float(
            np.linalg.norm(
                pitch_position
                -
                pending_pitch
            )
        )


        image_speed = (
            image_distance
            /
            dt
        )


        pitch_speed = (
            pitch_distance
            /
            dt
        )


        if (
            image_speed
            >
            REACQUIRE_MAX_IMAGE_SPEED_PX_S
        ):

            continue


        if (
            pitch_speed
            >
            REACQUIRE_MAX_PITCH_SPEED_MPS
        ):

            continue


        score = (

            0.55
            *
            (
                image_speed
                /
                REACQUIRE_MAX_IMAGE_SPEED_PX_S
            )

            +

            0.30
            *
            (
                pitch_speed
                /
                REACQUIRE_MAX_PITCH_SPEED_MPS
            )

            +

            0.15
            *
            (
                1.0
                -
                candidate["confidence"]
            )
        )


        options.append(
            (
                score,
                candidate
            )
        )


    if not options:

        return None


    options.sort(
        key=lambda item:
            item[0]
    )


    return options[0][1]


# ============================================================
# Video setup
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
# State
# ============================================================

last_H = None

fallback_streak = 0


history = deque(
    maxlen=2
)


trail = deque(
    maxlen=30
)


pending_reacquire = None


segment_id = -1


# ============================================================
# Statistics
# ============================================================

frame_idx = 0

raw_candidate_total = 0

pitch_rejected_total = 0

pitch_valid_total = 0

frames_with_raw = 0

frames_with_valid = 0

frames_with_multiple_valid = 0

accepted_frames = 0

missing_frames = 0

pending_frames = 0

reacquisition_count = 0

fallback_frames = 0


rows = []


print("")
print(
    "========================================"
)

print(
    "FORZASYS BALL TRACKING"
)

print(
    "Pitch ROI + Temporal Association"
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


# ============================================================
# Main loop
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    annotated = frame.copy()


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
    # 2. Visible pitch polygon
    # ========================================================

    pitch_polygon = (
        get_pitch_image_polygon(
            H
        )
    )


    # Draw pitch ROI
    if pitch_polygon is not None:

        polygon_int = (
            pitch_polygon
            .astype(np.int32)
            .reshape(-1, 1, 2)
        )


        cv2.polylines(
            annotated,
            [polygon_int],
            True,
            (
                0,
                255,
                0
            ),
            2
        )


    # ========================================================
    # 3. Ball candidates
    # ========================================================

    (
        raw_candidates,
        valid_candidates,
        rejected_candidates

    ) = detect_ball_candidates(
        frame,
        H,
        pitch_polygon
    )


    raw_count = len(
        raw_candidates
    )

    valid_count = len(
        valid_candidates
    )

    rejected_count = len(
        rejected_candidates
    )


    raw_candidate_total += raw_count

    pitch_valid_total += valid_count

    pitch_rejected_total += rejected_count


    if raw_count > 0:

        frames_with_raw += 1


    if valid_count > 0:

        frames_with_valid += 1


    if valid_count > 1:

        frames_with_multiple_valid += 1


    # ========================================================
    # 4. Draw pitch-rejected candidates
    #
    # RED = detector says ball,
    #       geometry says outside pitch.
    # ========================================================

    for candidate in rejected_candidates:

        x1, y1, x2, y2 = (
            candidate["box"]
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
                0,
                255
            ),
            2
        )


        cv2.putText(
            annotated,
            (
                f"REJECT OUTSIDE "
                f"{candidate['confidence']:.2f}"
            ),
            (
                int(x1),
                max(
                    int(y1) - 8,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Valid candidates
    #
    # CYAN = survives pitch geometry
    # ========================================================

    for candidate in valid_candidates:

        cv2.circle(
            annotated,
            (
                int(
                    candidate["image_x"]
                ),
                int(
                    candidate["image_y"]
                )
            ),
            5,
            (
                255,
                255,
                0
            ),
            1
        )


    # ========================================================
    # 5. Tracking state
    # ========================================================

    ball = None

    mode = "missing"

    estimated_speed = None

    image_residual = None

    pitch_residual = None


    # --------------------------------------------------------
    # Is track still alive?
    # --------------------------------------------------------

    active_track = False


    if len(history) > 0:

        gap_from_last = (

            frame_idx

            -

            int(
                history[-1]["frame"]
            )
        )


        active_track = (

            gap_from_last
            <=
            MAX_ASSOCIATION_GAP
        )


    # ========================================================
    # Active association
    # ========================================================

    if active_track:

        (
            ball,
            estimated_speed,
            image_residual,
            pitch_residual

        ) = associate_candidate(
            valid_candidates,
            history,
            frame_idx,
            fps
        )


        if ball is not None:

            mode = "tracked"

            pending_reacquire = None

        else:

            mode = "short_gap_missing"


    # ========================================================
    # Reacquisition
    # ========================================================

    else:

        confirmed = confirm_reacquisition(
            pending_reacquire,
            valid_candidates,
            frame_idx,
            fps
        )


        if confirmed is not None:

            ball = confirmed

            mode = "reacquired"

            segment_id += 1

            history.clear()
            trail.clear()

            pending_reacquire = None

            reacquisition_count += 1


        else:

            seed = choose_reacquisition_seed(
                valid_candidates
            )


            if seed is not None:

                pending_reacquire = {
                    **seed,
                    "frame":
                        frame_idx
                }


                mode = "pending"

                pending_frames += 1


            else:

                pending_reacquire = None

                mode = "waiting"


    # ========================================================
    # Accepted ball
    # ========================================================

    accepted = (
        ball is not None
    )


    if accepted:

        if segment_id < 0:

            segment_id = 0


        accepted_frames += 1


        history.append(
            {
                "frame":
                    frame_idx,

                "image_x":
                    ball["image_x"],

                "image_y":
                    ball["image_y"],

                "pitch_x_m":
                    ball["pitch_x_m"],

                "pitch_y_m":
                    ball["pitch_y_m"]
            }
        )


        trail.append(
            (
                frame_idx,
                ball["image_x"],
                ball["image_y"]
            )
        )


    else:

        missing_frames += 1


    # ========================================================
    # 6. Draw accepted ball
    #
    # YELLOW = final accepted position
    # ========================================================

    if accepted:

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
                int(
                    ball["image_x"]
                ),
                int(
                    ball["image_y"]
                )
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
                f"{ball['confidence']:.2f}"
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


    # ========================================================
    # Pending reacquisition
    #
    # MAGENTA = possible ball,
    #           not trusted yet.
    # ========================================================

    if (
        pending_reacquire is not None
        and
        not accepted
    ):

        cv2.circle(
            annotated,
            (
                int(
                    pending_reacquire[
                        "image_x"
                    ]
                ),
                int(
                    pending_reacquire[
                        "image_y"
                    ]
                )
            ),
            9,
            (
                255,
                0,
                255
            ),
            2
        )


        cv2.putText(
            annotated,
            "PENDING",
            (
                int(
                    pending_reacquire[
                        "image_x"
                    ]
                ) + 10,
                int(
                    pending_reacquire[
                        "image_y"
                    ]
                ) - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                255,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Trail
    # ========================================================

    trail_list = list(
        trail
    )


    for i in range(
        1,
        len(trail_list)
    ):

        previous = trail_list[
            i - 1
        ]

        current = trail_list[i]


        frame_gap = (
            current[0]
            -
            previous[0]
        )


        if (
            frame_gap
            <=
            MAX_ASSOCIATION_GAP
        ):

            cv2.line(
                annotated,
                (
                    int(previous[1]),
                    int(previous[2])
                ),
                (
                    int(current[1]),
                    int(current[2])
                ),
                (
                    0,
                    255,
                    255
                ),
                2
            )


    # ========================================================
    # Status
    # ========================================================

    cv2.putText(
        annotated,
        (
            f"ForzaSys Track | "
            f"{mode} | "
            f"segment={segment_id}"
        ),
        (
            20,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        annotated,
        (
            f"Raw={raw_count} | "
            f"Pitch-valid={valid_count} | "
            f"Outside rejected={rejected_count}"
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

            "raw_candidates":
                raw_count,

            "pitch_valid_candidates":
                valid_count,

            "pitch_rejected_candidates":
                rejected_count,

            "accepted":
                accepted,

            "mode":
                mode,

            "segment_id":
                (
                    segment_id
                    if accepted
                    else None
                ),

            "confidence":
                (
                    ball["confidence"]
                    if accepted
                    else None
                ),

            "image_x":
                (
                    ball["image_x"]
                    if accepted
                    else None
                ),

            "image_y":
                (
                    ball["image_y"]
                    if accepted
                    else None
                ),

            "pitch_x_m":
                (
                    ball["pitch_x_m"]
                    if accepted
                    else None
                ),

            "pitch_y_m":
                (
                    ball["pitch_y_m"]
                    if accepted
                    else None
                ),

            "estimated_speed_mps":
                estimated_speed,

            "image_residual_px":
                image_residual,

            "pitch_residual_m":
                pitch_residual,

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
    # Console
    # ========================================================

    if frame_idx % 30 == 0:

        print(
            f"frame={frame_idx:04d} | "
            f"raw={raw_count} | "
            f"valid={valid_count} | "
            f"reject={rejected_count} | "
            f"accepted={accepted} | "
            f"mode={mode}"
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

    "raw_candidates",
    "pitch_valid_candidates",
    "pitch_rejected_candidates",

    "accepted",
    "mode",
    "segment_id",

    "confidence",

    "image_x",
    "image_y",

    "pitch_x_m",
    "pitch_y_m",

    "estimated_speed_mps",

    "image_residual_px",
    "pitch_residual_m",

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
# Final statistics
# ============================================================

raw_coverage = (

    frames_with_raw
    /
    frame_idx
    *
    100.0

    if frame_idx > 0
    else 0.0
)


valid_coverage = (

    frames_with_valid
    /
    frame_idx
    *
    100.0

    if frame_idx > 0
    else 0.0
)


accepted_coverage = (

    accepted_frames
    /
    frame_idx
    *
    100.0

    if frame_idx > 0
    else 0.0
)


multiple_valid_rate = (

    frames_with_multiple_valid
    /
    frame_idx
    *
    100.0

    if frame_idx > 0
    else 0.0
)


# ============================================================
# Accepted speeds
# ============================================================

speeds = [

    row["estimated_speed_mps"]

    for row in rows

    if (
        row["estimated_speed_mps"]
        is not None
    )
]


speeds = np.asarray(
    speeds,
    dtype=np.float32
)


# ============================================================
# Print summary
# ============================================================

print("")
print(
    "========================================"
)

print(
    "FORZASYS BALL TRACKING SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    frame_idx
)

print("")

print(
    "Frames with raw ball detection:",
    frames_with_raw
)

print(
    "Raw detection coverage:",
    f"{raw_coverage:.1f}%"
)

print("")

print(
    "Raw candidates:",
    raw_candidate_total
)

print(
    "Pitch-valid candidates:",
    pitch_valid_total
)

print(
    "Pitch-rejected candidates:",
    pitch_rejected_total
)

print("")

print(
    "Frames with >=1 pitch-valid candidate:",
    frames_with_valid
)

print(
    "Pitch-valid coverage:",
    f"{valid_coverage:.1f}%"
)

print(
    "Frames with multiple pitch-valid candidates:",
    frames_with_multiple_valid
)

print(
    "Multiple pitch-valid rate:",
    f"{multiple_valid_rate:.1f}%"
)

print("")

print(
    "Accepted tracking frames:",
    accepted_frames
)

print(
    "Final tracking coverage:",
    f"{accepted_coverage:.1f}%"
)

print(
    "Missing frames:",
    missing_frames
)

print("")

print(
    "Pending frames:",
    pending_frames
)

print(
    "Confirmed reacquisitions:",
    reacquisition_count
)

print(
    "Trajectory segments:",
    (
        segment_id + 1
        if segment_id >= 0
        else 0
    )
)

print("")

print(
    "Homography fallback frames:",
    fallback_frames
)


if len(speeds) > 0:

    print("")

    print(
        "Speed median:",
        f"{np.median(speeds):.2f} m/s"
    )

    print(
        "Speed P95:",
        f"{np.percentile(speeds, 95):.2f} m/s"
    )

    print(
        "Speed P99:",
        f"{np.percentile(speeds, 99):.2f} m/s"
    )

    print(
        "Speed max:",
        f"{np.max(speeds):.2f} m/s"
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
