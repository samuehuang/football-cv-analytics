import os
import csv
from collections import deque

import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = "videos/match.mp4"

OLD_BALL_MODEL_PATH = "models/football-ball-detection.pt"
FORZA_MODEL_PATH = "models/yolov8m_forzasys_soccer.pt"
PITCH_MODEL_PATH = "models/football-pitch-detection.pt"

OUTPUT_VIDEO = "outputs/ball_tracking_ensemble.mp4"
OUTPUT_CSV = "outputs/ball_tracking_ensemble.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "mps"


# ============================================================
# MODELS
# ============================================================

old_ball_model = YOLO(OLD_BALL_MODEL_PATH)

forza_model = YOLO(FORZA_MODEL_PATH)

pitch_model = YOLO(PITCH_MODEL_PATH)


# ============================================================
# CLASS IDS
# ============================================================

# Old football model:
# BALL = 0

OLD_BALL_CLASS = 0


# ForzaSys:
# 0 player
# 1 ball
# 2 logo

FORZA_BALL_CLASS = 1


# ============================================================
# DETECTION
# ============================================================

OLD_CONF = 0.10
FORZA_CONF = 0.10

IMGSZ = 1280


# ============================================================
# CROSS MODEL MATCH
# ============================================================

# If centers from two independent models are within this
# distance, treat them as the same object.

DUAL_MATCH_DISTANCE_PX = 28.0


# ============================================================
# SINGLE MODEL RULES
# ============================================================

# During an ACTIVE track:
# a single-model candidate can still be used,
# but only if confidence is reasonably high.

SINGLE_TRACK_MIN_CONF = 0.40


# When completely reacquiring a lost ball,
# single-model evidence must be much stronger.

SINGLE_REACQUIRE_MIN_CONF = 0.70


# ============================================================
# HOMOGRAPHY
# ============================================================

CONFIG = SoccerPitchConfiguration()

pitch_vertices = np.asarray(
    CONFIG.vertices,
    dtype=np.float32
)


KP_CONF_THRESHOLD = 0.60

MIN_KEYPOINTS = 6
MIN_INLIER_RATIO = 0.60

RANSAC_THRESHOLD = 250.0

MAX_MEAN_REPROJECTION_ERROR = 30.0
MAX_MEDIAN_REPROJECTION_ERROR = 20.0

MAX_H_FALLBACK = 3


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


PITCH_MARGIN_CM = 75.0
IMAGE_PITCH_MARGIN_PX = 8.0


# ============================================================
# TEMPORAL ASSOCIATION
# ============================================================

MAX_ASSOCIATION_GAP = 5

MAX_BALL_SPEED_MPS = 65.0


BASE_IMAGE_GATE_PX = 110.0
IMAGE_GATE_PER_GAP = 65.0


BASE_PITCH_GATE_M = 7.0
PITCH_GATE_PER_GAP = 3.0


# Single candidates need to agree more closely with trajectory

SINGLE_IMAGE_GATE_SCALE = 0.55
SINGLE_PITCH_GATE_SCALE = 0.65


# ============================================================
# SCORING
# ============================================================

IMAGE_WEIGHT = 0.55
PITCH_WEIGHT = 0.25
CONF_WEIGHT = 0.10
MODEL_SUPPORT_WEIGHT = 0.10


# ============================================================
# REACQUISITION
# ============================================================

REACQUIRE_CONFIRM_MAX_GAP = 2

REACQUIRE_MAX_IMAGE_SPEED_PX_S = 2400.0
REACQUIRE_MAX_PITCH_SPEED_MPS = 65.0


# ============================================================
# HOMOGRAPHY
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


    if len(keypoints.xy) > 0:

        points = keypoints.xy[0]


        if keypoints.confidence is not None:

            confidence = keypoints.confidence[0]

        else:

            confidence = np.ones(
                len(points),
                dtype=np.float32
            )


        valid = (

            (points[:, 0] > 1)

            &

            (points[:, 1] > 1)

            &

            (
                confidence
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


        valid_kp_count = len(source)


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

                mask = (
                    inliers
                    .reshape(-1)
                    .astype(bool)
                )


                inlier_count = int(
                    np.sum(mask)
                )


                required = max(
                    5,
                    int(
                        np.ceil(
                            valid_kp_count
                            *
                            MIN_INLIER_RATIO
                        )
                    )
                )


                if inlier_count >= required:

                    try:

                        H_inv = np.linalg.inv(
                            H_candidate
                        )


                        target_inliers = (
                            target[mask]
                            .reshape(-1, 1, 2)
                        )


                        source_inliers = source[mask]


                        projected_back = (
                            cv2.perspectiveTransform(
                                target_inliers,
                                H_inv
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
                            <= MAX_MEAN_REPROJECTION_ERROR

                            and

                            median_error
                            <= MAX_MEDIAN_REPROJECTION_ERROR

                        ):

                            H = H_candidate

                            last_H = H_candidate

                            fallback_streak = 0


                    except (
                        np.linalg.LinAlgError,
                        cv2.error
                    ):

                        pass


    # Short homography fallback
    if H is None:

        if (

            last_H is not None

            and

            fallback_streak
            <
            MAX_H_FALLBACK

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
# PITCH IMAGE POLYGON
# ============================================================

def get_pitch_polygon(H):

    if H is None:

        return None


    try:

        H_inv = np.linalg.inv(H)

    except np.linalg.LinAlgError:

        return None


    corners = np.array(
        [
            [PITCH_X_MIN, PITCH_Y_MIN],
            [PITCH_X_MAX, PITCH_Y_MIN],
            [PITCH_X_MAX, PITCH_Y_MAX],
            [PITCH_X_MIN, PITCH_Y_MAX],
        ],
        dtype=np.float32
    )


    try:

        image_corners = (
            cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2),
                H_inv
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
# IMAGE -> PITCH
# ============================================================

def project_to_pitch(
    x,
    y,
    H
):

    if H is None:

        return None


    point = np.array(
        [[[
            x,
            y
        ]]],
        dtype=np.float32
    )


    try:

        result = cv2.perspectiveTransform(
            point,
            H
        )[0][0]

    except cv2.error:

        return None


    if not np.all(
        np.isfinite(result)
    ):

        return None


    return (
        float(result[0]),
        float(result[1])
    )


# ============================================================
# PITCH ROI TEST
# ============================================================

def inside_pitch(
    image_x,
    image_y,
    pitch_x_cm,
    pitch_y_cm,
    polygon
):

    if polygon is None:

        return False


    signed_distance = cv2.pointPolygonTest(

        polygon.reshape(
            -1,
            1,
            2
        ),

        (
            float(image_x),
            float(image_y)
        ),

        True
    )


    if (
        signed_distance
        <
        -IMAGE_PITCH_MARGIN_PX
    ):

        return False


    return (

        PITCH_X_MIN - PITCH_MARGIN_CM
        <=
        pitch_x_cm
        <=
        PITCH_X_MAX + PITCH_MARGIN_CM

        and

        PITCH_Y_MIN - PITCH_MARGIN_CM
        <=
        pitch_y_cm
        <=
        PITCH_Y_MAX + PITCH_MARGIN_CM
    )


# ============================================================
# RAW DETECTOR
# ============================================================

def detect_model(
    model,
    frame,
    class_id,
    confidence,
    model_name
):

    result = model(
        frame,
        imgsz=IMGSZ,
        conf=confidence,
        classes=[class_id],
        device=DEVICE,
        verbose=False
    )[0]


    boxes = result.boxes


    detections = []


    if (
        boxes is None
        or
        len(boxes) == 0
    ):

        return detections


    xyxy = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )


    confs = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )


    for box, conf in zip(
        xyxy,
        confs
    ):

        x1, y1, x2, y2 = box


        detections.append(
            {
                "box":
                    np.asarray(
                        box,
                        dtype=np.float32
                    ),

                "confidence":
                    float(conf),

                "image_x":
                    float(
                        (x1 + x2)
                        /
                        2.0
                    ),

                "image_y":
                    float(
                        (y1 + y2)
                        /
                        2.0
                    ),

                "model":
                    model_name
            }
        )


    return detections


# ============================================================
# CROSS MODEL ENSEMBLE
# ============================================================

def build_ensemble_candidates(
    old_detections,
    forza_detections
):

    candidates = []

    used_old = set()
    used_forza = set()


    matches = []


    # --------------------------------------------------------
    # Find every possible OLD ↔ FORZA match
    # --------------------------------------------------------

    for old_i, old in enumerate(
        old_detections
    ):

        for forza_i, forza in enumerate(
            forza_detections
        ):

            distance = float(
                np.hypot(
                    old["image_x"]
                    -
                    forza["image_x"],

                    old["image_y"]
                    -
                    forza["image_y"]
                )
            )


            if (
                distance
                <=
                DUAL_MATCH_DISTANCE_PX
            ):

                matches.append(
                    (
                        distance,
                        old_i,
                        forza_i
                    )
                )


    # Closest match first
    matches.sort(
        key=lambda x: x[0]
    )


    # --------------------------------------------------------
    # Greedy one-to-one matching
    # --------------------------------------------------------

    for distance, old_i, forza_i in matches:

        if old_i in used_old:

            continue


        if forza_i in used_forza:

            continue


        old = old_detections[old_i]
        forza = forza_detections[forza_i]


        used_old.add(old_i)
        used_forza.add(forza_i)


        # Confidence-weighted center
        w_old = max(
            old["confidence"],
            0.01
        )

        w_forza = max(
            forza["confidence"],
            0.01
        )


        total_weight = (
            w_old
            +
            w_forza
        )


        center_x = (

            old["image_x"]
            *
            w_old

            +

            forza["image_x"]
            *
            w_forza

        ) / total_weight


        center_y = (

            old["image_y"]
            *
            w_old

            +

            forza["image_y"]
            *
            w_forza

        ) / total_weight


        # Keep box from stronger detector
        if (
            old["confidence"]
            >=
            forza["confidence"]
        ):

            box = old["box"]

        else:

            box = forza["box"]


        candidates.append(
            {
                "type":
                    "dual",

                "box":
                    box,

                "image_x":
                    float(center_x),

                "image_y":
                    float(center_y),

                "confidence":
                    float(
                        max(
                            old["confidence"],
                            forza["confidence"]
                        )
                    ),

                "old_conf":
                    old["confidence"],

                "forza_conf":
                    forza["confidence"],

                "match_distance_px":
                    distance
            }
        )


    # --------------------------------------------------------
    # OLD-only
    # --------------------------------------------------------

    for old_i, old in enumerate(
        old_detections
    ):

        if old_i in used_old:

            continue


        candidates.append(
            {
                "type":
                    "old_single",

                "box":
                    old["box"],

                "image_x":
                    old["image_x"],

                "image_y":
                    old["image_y"],

                "confidence":
                    old["confidence"],

                "old_conf":
                    old["confidence"],

                "forza_conf":
                    None,

                "match_distance_px":
                    None
            }
        )


    # --------------------------------------------------------
    # FORZA-only
    # --------------------------------------------------------

    for forza_i, forza in enumerate(
        forza_detections
    ):

        if forza_i in used_forza:

            continue


        candidates.append(
            {
                "type":
                    "forza_single",

                "box":
                    forza["box"],

                "image_x":
                    forza["image_x"],

                "image_y":
                    forza["image_y"],

                "confidence":
                    forza["confidence"],

                "old_conf":
                    None,

                "forza_conf":
                    forza["confidence"],

                "match_distance_px":
                    None
            }
        )


    return candidates


# ============================================================
# PROJECT + PITCH FILTER
# ============================================================

def pitch_filter_candidates(
    candidates,
    H,
    polygon
):

    valid = []
    rejected = []


    for candidate in candidates:

        projection = project_to_pitch(
            candidate["image_x"],
            candidate["image_y"],
            H
        )


        if projection is None:

            rejected.append(
                candidate
            )

            continue


        x_cm, y_cm = projection


        candidate = candidate.copy()


        candidate["pitch_x_cm"] = x_cm
        candidate["pitch_y_cm"] = y_cm

        candidate["pitch_x_m"] = (
            x_cm / 100.0
        )

        candidate["pitch_y_m"] = (
            y_cm / 100.0
        )


        if inside_pitch(
            candidate["image_x"],
            candidate["image_y"],
            x_cm,
            y_cm,
            polygon
        ):

            valid.append(
                candidate
            )

        else:

            rejected.append(
                candidate
            )


    return (
        valid,
        rejected
    )


# ============================================================
# PREDICTION
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
        last["frame"]
    )


    dt = frame_gap / fps


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
            previous_gap <= MAX_ASSOCIATION_GAP
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
                last_image
                -
                previous_image
            ) / previous_dt


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
                last_pitch
                -
                previous_pitch
            ) / previous_dt


            if (
                np.linalg.norm(
                    pitch_velocity
                )
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
# ACTIVE ASSOCIATION
# ============================================================

def associate_active(
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
            None,
            None
        )


    last_pitch = np.array(
        [
            history[-1]["pitch_x_m"],
            history[-1]["pitch_y_m"]
        ],
        dtype=np.float32
    )


    base_image_gate = (

        BASE_IMAGE_GATE_PX

        +

        IMAGE_GATE_PER_GAP
        *
        (
            frame_gap - 1
        )
    )


    base_pitch_gate = (

        BASE_PITCH_GATE_M

        +

        PITCH_GATE_PER_GAP
        *
        (
            frame_gap - 1
        )
    )


    options = []


    for candidate in candidates:

        is_dual = (
            candidate["type"]
            ==
            "dual"
        )


        # Single candidate confidence requirement
        if (
            not is_dual
            and
            candidate["confidence"]
            <
            SINGLE_TRACK_MIN_CONF
        ):

            continue


        image_gate = (
            base_image_gate

            if is_dual

            else

            base_image_gate
            *
            SINGLE_IMAGE_GATE_SCALE
        )


        pitch_gate = (
            base_pitch_gate

            if is_dual

            else

            base_pitch_gate
            *
            SINGLE_PITCH_GATE_SCALE
        )


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


        speed = (
            distance_from_last
            /
            dt
        )


        if speed > MAX_BALL_SPEED_MPS:

            continue


        if image_residual > image_gate:

            continue


        if pitch_residual > pitch_gate:

            continue


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


        support_cost = (
            0.0
            if is_dual
            else
            1.0
        )


        score = (

            IMAGE_WEIGHT
            *
            image_cost

            +

            PITCH_WEIGHT
            *
            pitch_cost

            +

            CONF_WEIGHT
            *
            confidence_cost

            +

            MODEL_SUPPORT_WEIGHT
            *
            support_cost
        )


        options.append(
            (
                score,
                candidate,
                speed,
                image_residual,
                pitch_residual
            )
        )


    if not options:

        return (
            None,
            None,
            None,
            None,
            None
        )


    options.sort(
        key=lambda x: x[0]
    )


    (
        score,
        candidate,
        speed,
        image_residual,
        pitch_residual

    ) = options[0]


    return (
        candidate,
        float(score),
        float(speed),
        float(image_residual),
        float(pitch_residual)
    )


# ============================================================
# REACQUISITION ELIGIBILITY
# ============================================================

def reacquire_eligible(
    candidate
):

    if candidate["type"] == "dual":

        return True


    return (
        candidate["confidence"]
        >=
        SINGLE_REACQUIRE_MIN_CONF
    )


# ============================================================
# REACQUISITION SEED
# ============================================================

def choose_seed(
    candidates
):

    eligible = [

        c

        for c in candidates

        if reacquire_eligible(c)
    ]


    if not eligible:

        return None


    # DUAL always preferred over SINGLE
    eligible.sort(
        key=lambda c:
        (
            0
            if c["type"] == "dual"
            else 1,

            -c["confidence"]
        )
    )


    return eligible[0]


# ============================================================
# CONFIRM REACQUISITION
# ============================================================

def confirm_seed(
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
        frame_gap
        >
        REACQUIRE_CONFIRM_MAX_GAP
    ):

        return None


    dt = (
        frame_gap
        /
        fps
    )


    old_image = np.array(
        [
            pending["image_x"],
            pending["image_y"]
        ],
        dtype=np.float32
    )


    old_pitch = np.array(
        [
            pending["pitch_x_m"],
            pending["pitch_y_m"]
        ],
        dtype=np.float32
    )


    options = []


    for candidate in candidates:

        if not reacquire_eligible(
            candidate
        ):

            continue


        image = np.array(
            [
                candidate["image_x"],
                candidate["image_y"]
            ],
            dtype=np.float32
        )


        pitch = np.array(
            [
                candidate["pitch_x_m"],
                candidate["pitch_y_m"]
            ],
            dtype=np.float32
        )


        image_speed = float(
            np.linalg.norm(
                image
                -
                old_image
            )
            /
            dt
        )


        pitch_speed = float(
            np.linalg.norm(
                pitch
                -
                old_pitch
            )
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


        support_penalty = (

            0.0

            if candidate["type"] == "dual"

            else

            1.0
        )


        score = (

            0.50
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

            0.10
            *
            (
                1.0
                -
                candidate["confidence"]
            )

            +

            0.10
            *
            support_penalty
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
        key=lambda x: x[0]
    )


    return options[0][1]


# ============================================================
# VIDEO
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


# ============================================================
# STATE
# ============================================================

last_H = None
fallback_streak = 0

history = deque(
    maxlen=2
)

trail = deque(
    maxlen=30
)

pending = None

segment_id = -1

rows = []


# ============================================================
# STATISTICS
# ============================================================

frame_idx = 0

accepted_frames = 0
dual_accepted = 0
single_accepted = 0

missing_frames = 0

reacquisitions = 0
pending_frames = 0

old_raw_total = 0
forza_raw_total = 0

dual_candidate_total = 0
single_candidate_total = 0

pitch_rejected_total = 0

accepted_speeds = []


print("")
print("========================================")
print("BALL TRACKING - CROSS MODEL ENSEMBLE")
print("========================================")
print("")

print("Frames:", total_frames)
print("FPS:", fps)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    annotated = frame.copy()


    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    (
        H,
        last_H,
        fallback_streak,
        homography_fallback,
        valid_kp,
        inliers,
        mean_error,
        median_error

    ) = estimate_homography(
        frame,
        last_H,
        fallback_streak
    )


    polygon = get_pitch_polygon(
        H
    )


    # --------------------------------------------------------
    # Both detectors
    # --------------------------------------------------------

    old_raw = detect_model(
        old_ball_model,
        frame,
        OLD_BALL_CLASS,
        OLD_CONF,
        "old"
    )


    forza_raw = detect_model(
        forza_model,
        frame,
        FORZA_BALL_CLASS,
        FORZA_CONF,
        "forza"
    )


    old_raw_total += len(
        old_raw
    )

    forza_raw_total += len(
        forza_raw
    )


    ensemble = build_ensemble_candidates(
        old_raw,
        forza_raw
    )


    dual_count = sum(
        c["type"] == "dual"
        for c in ensemble
    )


    single_count = (
        len(ensemble)
        -
        dual_count
    )


    dual_candidate_total += dual_count
    single_candidate_total += single_count


    (
        candidates,
        rejected

    ) = pitch_filter_candidates(
        ensemble,
        H,
        polygon
    )


    pitch_rejected_total += len(
        rejected
    )


    # --------------------------------------------------------
    # Draw candidates
    #
    # Cyan = DUAL
    # Gray = SINGLE
    # --------------------------------------------------------

    for candidate in candidates:

        x = int(
            candidate["image_x"]
        )

        y = int(
            candidate["image_y"]
        )


        if candidate["type"] == "dual":

            color = (
                255,
                255,
                0
            )

            label = "DUAL"

        else:

            color = (
                140,
                140,
                140
            )

            label = "SINGLE"


        cv2.circle(
            annotated,
            (x, y),
            5,
            color,
            1
        )


        cv2.putText(
            annotated,
            label,
            (
                x + 6,
                y - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Track state
    # --------------------------------------------------------

    ball = None

    mode = "missing"

    score = None
    speed = None

    image_residual = None
    pitch_residual = None


    active = False


    if len(history) > 0:

        gap = (
            frame_idx
            -
            history[-1]["frame"]
        )


        active = (
            gap
            <=
            MAX_ASSOCIATION_GAP
        )


    # --------------------------------------------------------
    # Active tracking
    # --------------------------------------------------------

    if active:

        (
            ball,
            score,
            speed,
            image_residual,
            pitch_residual

        ) = associate_active(
            candidates,
            history,
            frame_idx,
            fps
        )


        if ball is not None:

            mode = "tracked"

            pending = None

        else:

            mode = "short_gap_missing"


    # --------------------------------------------------------
    # Reacquisition
    # --------------------------------------------------------

    else:

        confirmed = confirm_seed(
            pending,
            candidates,
            frame_idx,
            fps
        )


        if confirmed is not None:

            ball = confirmed

            mode = "reacquired"

            segment_id += 1

            history.clear()
            trail.clear()

            pending = None

            reacquisitions += 1


        else:

            seed = choose_seed(
                candidates
            )


            if seed is not None:

                pending = {
                    **seed,
                    "frame":
                        frame_idx
                }


                mode = "pending"

                pending_frames += 1

            else:

                pending = None

                mode = "waiting"


    # --------------------------------------------------------
    # Accepted
    # --------------------------------------------------------

    accepted = (
        ball is not None
    )


    if accepted:

        if segment_id < 0:

            segment_id = 0


        accepted_frames += 1


        if ball["type"] == "dual":

            dual_accepted += 1

        else:

            single_accepted += 1


        if speed is not None:

            accepted_speeds.append(
                speed
            )


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


    # --------------------------------------------------------
    # Draw accepted ball
    # --------------------------------------------------------

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
                int(ball["image_x"]),
                int(ball["image_y"])
            ),
            8,
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
                f"{ball['type'].upper()} "
                f"{ball['confidence']:.2f}"
            ),
            (
                int(x1),
                max(
                    20,
                    int(y1) - 10
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


    # --------------------------------------------------------
    # Pending
    # --------------------------------------------------------

    if (
        pending is not None
        and
        not accepted
    ):

        cv2.circle(
            annotated,
            (
                int(pending["image_x"]),
                int(pending["image_y"])
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
                int(pending["image_x"]) + 10,
                int(pending["image_y"]) - 8
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


    # --------------------------------------------------------
    # Trail
    # --------------------------------------------------------

    trail_list = list(
        trail
    )


    for i in range(
        1,
        len(trail_list)
    ):

        a = trail_list[i - 1]
        b = trail_list[i]


        if (
            b[0] - a[0]
            <=
            MAX_ASSOCIATION_GAP
        ):

            cv2.line(
                annotated,
                (
                    int(a[1]),
                    int(a[2])
                ),
                (
                    int(b[1]),
                    int(b[2])
                ),
                (
                    0,
                    255,
                    255
                ),
                2
            )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    cv2.putText(
        annotated,
        (
            f"Ensemble | "
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
            f"Old={len(old_raw)} | "
            f"Forza={len(forza_raw)} | "
            f"Dual={dual_count} | "
            f"Single={single_count} | "
            f"Pitch-valid={len(candidates)}"
        ),
        (
            20,
            65
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    rows.append(
        {
            "frame":
                frame_idx,

            "time_sec":
                frame_idx / fps,

            "old_raw_candidates":
                len(old_raw),

            "forza_raw_candidates":
                len(forza_raw),

            "dual_candidates":
                dual_count,

            "single_candidates":
                single_count,

            "pitch_valid_candidates":
                len(candidates),

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

            "candidate_type":
                (
                    ball["type"]
                    if accepted
                    else None
                ),

            "confidence":
                (
                    ball["confidence"]
                    if accepted
                    else None
                ),

            "old_conf":
                (
                    ball["old_conf"]
                    if accepted
                    else None
                ),

            "forza_conf":
                (
                    ball["forza_conf"]
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
                speed,

            "association_score":
                score,

            "image_residual_px":
                image_residual,

            "pitch_residual_m":
                pitch_residual,

            "valid_keypoints":
                valid_kp,

            "inliers":
                inliers,

            "mean_reprojection_error_px":
                mean_error,

            "median_reprojection_error_px":
                median_error,

            "homography_fallback":
                homography_fallback
        }
    )


    if frame_idx % 30 == 0:

        print(
            f"frame={frame_idx:04d} | "
            f"old={len(old_raw)} | "
            f"forza={len(forza_raw)} | "
            f"dual={dual_count} | "
            f"single={single_count} | "
            f"accepted={accepted} | "
            f"mode={mode}"
        )


    writer.write(
        annotated
    )


    frame_idx += 1


# ============================================================
# FINISH
# ============================================================

cap.release()
writer.release()


# ============================================================
# CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = list(
        rows[0].keys()
    )


    csv_writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )


    csv_writer.writeheader()

    csv_writer.writerows(
        rows
    )


# ============================================================
# SUMMARY
# ============================================================

coverage = (

    accepted_frames
    /
    frame_idx
    *
    100.0

    if frame_idx > 0

    else 0.0
)


dual_share = (

    dual_accepted
    /
    accepted_frames
    *
    100.0

    if accepted_frames > 0

    else 0.0
)


speeds = np.asarray(
    accepted_speeds,
    dtype=np.float32
)


print("")
print("========================================")
print("ENSEMBLE BALL TRACKING SUMMARY")
print("========================================")
print("")

print("Frames:", frame_idx)

print("")

print(
    "Old detector candidates:",
    old_raw_total
)

print(
    "Forza detector candidates:",
    forza_raw_total
)

print("")

print(
    "Dual-model candidates:",
    dual_candidate_total
)

print(
    "Single-model candidates:",
    single_candidate_total
)

print(
    "Pitch rejected:",
    pitch_rejected_total
)

print("")

print(
    "Accepted frames:",
    accepted_frames
)

print(
    "Tracking coverage:",
    f"{coverage:.1f}%"
)

print(
    "Missing frames:",
    missing_frames
)

print("")

print(
    "Dual accepted:",
    dual_accepted
)

print(
    "Single accepted:",
    single_accepted
)

print(
    "Dual share:",
    f"{dual_share:.1f}%"
)

print("")

print(
    "Pending frames:",
    pending_frames
)

print(
    "Confirmed reacquisitions:",
    reacquisitions
)

print(
    "Trajectory segments:",
    (
        segment_id + 1
        if segment_id >= 0
        else 0
    )
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
