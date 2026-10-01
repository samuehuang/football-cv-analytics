#!/usr/bin/env python3

import argparse
import csv
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration


# ============================================================
# Validated algorithm parameters
# ============================================================

OLD_BALL_CLASS = 0
FORZA_BALL_CLASS = 1

OLD_CONF = 0.10
FORZA_CONF = 0.10

IMGSZ = 1280


# ============================================================
# Cross-model ensemble
# ============================================================

DUAL_MATCH_DISTANCE_PX = 28.0


# ============================================================
# Single-model tracking
# ============================================================

SINGLE_TRACK_MIN_CONF = 0.40

SINGLE_REACQUIRE_MIN_CONF = 0.70


# ============================================================
# Homography
# ============================================================

KP_CONF_THRESHOLD = 0.60

MIN_KEYPOINTS = 6

MIN_INLIER_RATIO = 0.60

RANSAC_THRESHOLD = 250.0

MAX_MEAN_REPROJECTION_ERROR = 30.0

MAX_MEDIAN_REPROJECTION_ERROR = 20.0

MAX_H_FALLBACK = 3


PITCH_MARGIN_CM = 75.0

IMAGE_PITCH_MARGIN_PX = 8.0


# ============================================================
# Temporal association
# ============================================================

MAX_ASSOCIATION_GAP = 5

MAX_BALL_SPEED_MPS = 65.0


BASE_IMAGE_GATE_PX = 110.0

IMAGE_GATE_PER_GAP = 65.0


BASE_PITCH_GATE_M = 7.0

PITCH_GATE_PER_GAP = 3.0


SINGLE_IMAGE_GATE_SCALE = 0.55

SINGLE_PITCH_GATE_SCALE = 0.65


# ============================================================
# Association score
# ============================================================

IMAGE_WEIGHT = 0.55

PITCH_WEIGHT = 0.25

CONF_WEIGHT = 0.10

MODEL_SUPPORT_WEIGHT = 0.10


# ============================================================
# Reacquisition
# ============================================================

REACQUIRE_CONFIRM_MAX_GAP = 2

REACQUIRE_MAX_IMAGE_SPEED_PX_S = 2400.0

REACQUIRE_MAX_PITCH_SPEED_MPS = 65.0


# ============================================================
# Trusted layer
# ============================================================

TRUST_SINGLE_MIN_CONF = 0.50

TRUST_SINGLE_MAX_IMAGE_RESIDUAL_PX = 55.0

TRUST_SINGLE_MAX_PITCH_RESIDUAL_M = 3.5

TRUST_SINGLE_MAX_SPEED_MPS = 50.0


TRUST_SINGLE_REACQUIRE_MIN_CONF = 0.70


MAX_INTERPOLATION_GAP = 3

MAX_INTERPOLATION_SPEED_MPS = 35.0


# ============================================================
# Pitch configuration
# ============================================================

CONFIG = SoccerPitchConfiguration()


PITCH_VERTICES = np.asarray(
    CONFIG.vertices,
    dtype=np.float32,
)


PITCH_X_MIN = float(
    np.min(
        PITCH_VERTICES[:, 0]
    )
)


PITCH_X_MAX = float(
    np.max(
        PITCH_VERTICES[:, 0]
    )
)


PITCH_Y_MIN = float(
    np.min(
        PITCH_VERTICES[:, 1]
    )
)


PITCH_Y_MAX = float(
    np.max(
        PITCH_VERTICES[:, 1]
    )
)


# ============================================================
# Runtime globals
# ============================================================

DEVICE = "cpu"

old_ball_model = None

forza_model = None

pitch_model = None


# ============================================================
# General helpers
# ============================================================

def resolve_device(
    requested,
):

    if requested != "auto":

        return requested


    try:

        import torch


        if torch.cuda.is_available():

            return "0"


        if (
            hasattr(
                torch.backends,
                "mps",
            )
            and
            torch.backends.mps.is_available()
        ):

            return "mps"


    except Exception:

        pass


    return "cpu"


def require_file(
    path,
    label,
):

    path = Path(
        path
    )


    if not path.is_file():

        raise FileNotFoundError(
            f"Missing {label}: "
            f"{path}"
        )


def ensure_parent(
    path,
):

    Path(
        path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def get_video_fps(
    video_path,
):

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    cap.release()


    if fps <= 0:

        raise RuntimeError(
            "Invalid video FPS."
        )


    return float(
        fps
    )


# ============================================================
# Homography
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
        sv.KeyPoints
        .from_ultralytics(
            result
        )
    )


    H = None

    valid_kp_count = 0

    inlier_count = 0

    mean_error = None

    median_error = None

    used_fallback = False


    if len(
        keypoints.xy
    ) > 0:

        points = (
            keypoints.xy[0]
        )


        if (
            keypoints
            .keypoint_confidence
            is not None
        ):

            confidence = (
                keypoints
                .keypoint_confidence[0]
            )

        else:

            confidence = np.ones(

                len(
                    points
                ),

                dtype=np.float32,

            )


        valid = (

            (
                points[:, 0]
                >
                1
            )

            &

            (
                points[:, 1]
                >
                1
            )

            &

            (
                confidence
                >=
                KP_CONF_THRESHOLD
            )

        )


        source = (

            points[
                valid
            ]

            .astype(
                np.float32
            )

        )


        target = (

            PITCH_VERTICES[
                valid
            ]

            .astype(
                np.float32
            )

        )


        valid_kp_count = len(
            source
        )


        if (
            valid_kp_count
            >=
            MIN_KEYPOINTS
        ):

            (
                H_candidate,
                inliers,
            ) = cv2.findHomography(

                source,

                target,

                cv2.RANSAC,

                RANSAC_THRESHOLD,

            )


            if (
                H_candidate
                is not None
                and
                inliers
                is not None
            ):

                mask = (

                    inliers

                    .reshape(
                        -1
                    )

                    .astype(
                        bool
                    )

                )


                inlier_count = int(
                    np.sum(
                        mask
                    )
                )


                required = max(

                    5,

                    int(
                        np.ceil(
                            valid_kp_count
                            *
                            MIN_INLIER_RATIO
                        )
                    ),

                )


                if (
                    inlier_count
                    >=
                    required
                ):

                    try:

                        H_inv = (
                            np.linalg.inv(
                                H_candidate
                            )
                        )


                        target_inliers = (

                            target[
                                mask
                            ]

                            .reshape(
                                -1,
                                1,
                                2,
                            )

                        )


                        source_inliers = (
                            source[
                                mask
                            ]
                        )


                        projected_back = (

                            cv2
                            .perspectiveTransform(

                                target_inliers,

                                H_inv,

                            )

                            .reshape(
                                -1,
                                2,
                            )

                        )


                        errors = np.linalg.norm(

                            projected_back
                            -
                            source_inliers,

                            axis=1,

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
                            <=
                            MAX_MEAN_REPROJECTION_ERROR

                            and

                            median_error
                            <=
                            MAX_MEDIAN_REPROJECTION_ERROR

                        ):

                            H = (
                                H_candidate
                            )


                            last_H = (
                                H_candidate
                            )


                            fallback_streak = 0


                    except (
                        np.linalg.LinAlgError,
                        cv2.error,
                    ):

                        pass


    if H is None:

        if (

            last_H
            is not None

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

        median_error,

    )


# ============================================================
# Pitch polygon
# ============================================================

def get_pitch_polygon(
    H,
):

    if H is None:

        return None


    try:

        H_inv = (
            np.linalg.inv(
                H
            )
        )


    except np.linalg.LinAlgError:

        return None


    corners = np.array(

        [
            [
                PITCH_X_MIN,
                PITCH_Y_MIN,
            ],

            [
                PITCH_X_MAX,
                PITCH_Y_MIN,
            ],

            [
                PITCH_X_MAX,
                PITCH_Y_MAX,
            ],

            [
                PITCH_X_MIN,
                PITCH_Y_MAX,
            ],
        ],

        dtype=np.float32,

    )


    try:

        image_corners = (

            cv2
            .perspectiveTransform(

                corners.reshape(
                    -1,
                    1,
                    2,
                ),

                H_inv,

            )

            .reshape(
                -1,
                2,
            )

        )


    except cv2.error:

        return None


    if not np.all(
        np.isfinite(
            image_corners
        )
    ):

        return None


    return image_corners.astype(
        np.float32
    )


# ============================================================
# Image -> pitch
# ============================================================

def project_to_pitch(
    x,
    y,
    H,
):

    if H is None:

        return None


    point = np.array(

        [
            [
                [
                    x,
                    y,
                ]
            ]
        ],

        dtype=np.float32,

    )


    try:

        result = (
            cv2
            .perspectiveTransform(
                point,
                H,
            )[0][0]
        )


    except cv2.error:

        return None


    if not np.all(
        np.isfinite(
            result
        )
    ):

        return None


    return (
        float(
            result[0]
        ),
        float(
            result[1]
        ),
    )


# ============================================================
# Pitch ROI
# ============================================================

def inside_pitch(
    image_x,
    image_y,
    pitch_x_cm,
    pitch_y_cm,
    polygon,
):

    if polygon is None:

        return False


    signed_distance = (
        cv2.pointPolygonTest(

            polygon.reshape(
                -1,
                1,
                2,
            ),

            (
                float(
                    image_x
                ),

                float(
                    image_y
                ),
            ),

            True,

        )
    )


    if (
        signed_distance
        <
        -IMAGE_PITCH_MARGIN_PX
    ):

        return False


    return (

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


# ============================================================
# Raw detector
# ============================================================

def detect_model(
    model,
    frame,
    class_id,
    confidence,
    model_name,
):

    result = model(

        frame,

        imgsz=IMGSZ,

        conf=confidence,

        classes=[
            class_id
        ],

        device=DEVICE,

        verbose=False,

    )[0]


    boxes = (
        result.boxes
    )


    detections = []


    if (
        boxes is None
        or
        len(
            boxes
        ) == 0
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


    for (
        box,
        conf,
    ) in zip(
        xyxy,
        confs,
    ):

        (
            x1,
            y1,
            x2,
            y2,
        ) = box


        detections.append(
            {

                "box":
                    np.asarray(
                        box,
                        dtype=np.float32,
                    ),

                "confidence":
                    float(
                        conf
                    ),

                "image_x":
                    float(
                        (
                            x1
                            +
                            x2
                        )
                        /
                        2.0
                    ),

                "image_y":
                    float(
                        (
                            y1
                            +
                            y2
                        )
                        /
                        2.0
                    ),

                "model":
                    model_name,

            }
        )


    return detections


# ============================================================
# Cross-model candidate construction
# ============================================================

def build_ensemble_candidates(
    old_detections,
    forza_detections,
):

    candidates = []

    used_old = set()

    used_forza = set()

    matches = []


    for (
        old_i,
        old,
    ) in enumerate(
        old_detections
    ):

        for (
            forza_i,
            forza,
        ) in enumerate(
            forza_detections
        ):

            distance = float(

                np.hypot(

                    old[
                        "image_x"
                    ]
                    -
                    forza[
                        "image_x"
                    ],

                    old[
                        "image_y"
                    ]
                    -
                    forza[
                        "image_y"
                    ],

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
                        forza_i,
                    )
                )


    matches.sort(
        key=lambda x:
            x[0]
    )


    for (
        distance,
        old_i,
        forza_i,
    ) in matches:

        if (
            old_i
            in
            used_old
        ):

            continue


        if (
            forza_i
            in
            used_forza
        ):

            continue


        old = (
            old_detections[
                old_i
            ]
        )


        forza = (
            forza_detections[
                forza_i
            ]
        )


        used_old.add(
            old_i
        )


        used_forza.add(
            forza_i
        )


        w_old = max(

            old[
                "confidence"
            ],

            0.01,

        )


        w_forza = max(

            forza[
                "confidence"
            ],

            0.01,

        )


        total_weight = (
            w_old
            +
            w_forza
        )


        center_x = (

            old[
                "image_x"
            ]
            *
            w_old

            +

            forza[
                "image_x"
            ]
            *
            w_forza

        ) / total_weight


        center_y = (

            old[
                "image_y"
            ]
            *
            w_old

            +

            forza[
                "image_y"
            ]
            *
            w_forza

        ) / total_weight


        if (
            old[
                "confidence"
            ]
            >=
            forza[
                "confidence"
            ]
        ):

            box = (
                old[
                    "box"
                ]
            )

        else:

            box = (
                forza[
                    "box"
                ]
            )


        candidates.append(
            {

                "type":
                    "dual",

                "box":
                    box,

                "image_x":
                    float(
                        center_x
                    ),

                "image_y":
                    float(
                        center_y
                    ),

                "confidence":
                    float(
                        max(

                            old[
                                "confidence"
                            ],

                            forza[
                                "confidence"
                            ],

                        )
                    ),

                "old_conf":
                    old[
                        "confidence"
                    ],

                "forza_conf":
                    forza[
                        "confidence"
                    ],

                "match_distance_px":
                    distance,

            }
        )


    for (
        old_i,
        old,
    ) in enumerate(
        old_detections
    ):

        if (
            old_i
            in
            used_old
        ):

            continue


        candidates.append(
            {

                "type":
                    "old_single",

                "box":
                    old[
                        "box"
                    ],

                "image_x":
                    old[
                        "image_x"
                    ],

                "image_y":
                    old[
                        "image_y"
                    ],

                "confidence":
                    old[
                        "confidence"
                    ],

                "old_conf":
                    old[
                        "confidence"
                    ],

                "forza_conf":
                    None,

                "match_distance_px":
                    None,

            }
        )


    for (
        forza_i,
        forza,
    ) in enumerate(
        forza_detections
    ):

        if (
            forza_i
            in
            used_forza
        ):

            continue


        candidates.append(
            {

                "type":
                    "forza_single",

                "box":
                    forza[
                        "box"
                    ],

                "image_x":
                    forza[
                        "image_x"
                    ],

                "image_y":
                    forza[
                        "image_y"
                    ],

                "confidence":
                    forza[
                        "confidence"
                    ],

                "old_conf":
                    None,

                "forza_conf":
                    forza[
                        "confidence"
                    ],

                "match_distance_px":
                    None,

            }
        )


    return candidates


# ============================================================
# Pitch filter
# ============================================================

def pitch_filter_candidates(
    candidates,
    H,
    polygon,
):

    valid = []

    rejected = []


    for candidate in (
        candidates
    ):

        projection = (
            project_to_pitch(

                candidate[
                    "image_x"
                ],

                candidate[
                    "image_y"
                ],

                H,

            )
        )


        if projection is None:

            rejected.append(
                candidate
            )

            continue


        (
            x_cm,
            y_cm,
        ) = projection


        candidate = (
            candidate.copy()
        )


        candidate[
            "pitch_x_cm"
        ] = x_cm


        candidate[
            "pitch_y_cm"
        ] = y_cm


        candidate[
            "pitch_x_m"
        ] = (
            x_cm
            /
            100.0
        )


        candidate[
            "pitch_y_m"
        ] = (
            y_cm
            /
            100.0
        )


        if inside_pitch(

            candidate[
                "image_x"
            ],

            candidate[
                "image_y"
            ],

            x_cm,

            y_cm,

            polygon,

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
        rejected,
    )


# ============================================================
# Position prediction
# ============================================================

def predict_position(
    history,
    frame_idx,
    fps,
):

    last = (
        history[-1]
    )


    frame_gap = (

        frame_idx

        -

        last[
            "frame"
        ]

    )


    dt = (
        frame_gap
        /
        fps
    )


    predicted_image = np.array(

        [
            last[
                "image_x"
            ],

            last[
                "image_y"
            ],
        ],

        dtype=np.float32,

    )


    predicted_pitch = np.array(

        [
            last[
                "pitch_x_m"
            ],

            last[
                "pitch_y_m"
            ],
        ],

        dtype=np.float32,

    )


    if len(
        history
    ) >= 2:

        previous = (
            history[-2]
        )


        previous_gap = (

            last[
                "frame"
            ]

            -

            previous[
                "frame"
            ]

        )


        previous_dt = (
            previous_gap
            /
            fps
        )


        if (

            previous_dt
            >
            0

            and

            previous_gap
            <=
            MAX_ASSOCIATION_GAP

        ):

            previous_image = np.array(

                [
                    previous[
                        "image_x"
                    ],

                    previous[
                        "image_y"
                    ],
                ],

                dtype=np.float32,

            )


            last_image = np.array(

                [
                    last[
                        "image_x"
                    ],

                    last[
                        "image_y"
                    ],
                ],

                dtype=np.float32,

            )


            image_velocity = (

                last_image

                -

                previous_image

            ) / previous_dt


            previous_pitch = np.array(

                [
                    previous[
                        "pitch_x_m"
                    ],

                    previous[
                        "pitch_y_m"
                    ],
                ],

                dtype=np.float32,

            )


            last_pitch = np.array(

                [
                    last[
                        "pitch_x_m"
                    ],

                    last[
                        "pitch_y_m"
                    ],
                ],

                dtype=np.float32,

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

        predicted_pitch,

    )


# ============================================================
# Active temporal association
# ============================================================

def associate_active(
    candidates,
    history,
    frame_idx,
    fps,
):

    if len(
        history
    ) == 0:

        return (
            None,
            None,
            None,
            None,
            None,
        )


    (
        frame_gap,
        dt,
        predicted_image,
        predicted_pitch,
    ) = predict_position(

        history,

        frame_idx,

        fps,

    )


    if (

        frame_gap
        <=
        0

        or

        frame_gap
        >
        MAX_ASSOCIATION_GAP

    ):

        return (
            None,
            None,
            None,
            None,
            None,
        )


    last_pitch = np.array(

        [
            history[-1][
                "pitch_x_m"
            ],

            history[-1][
                "pitch_y_m"
            ],
        ],

        dtype=np.float32,

    )


    base_image_gate = (

        BASE_IMAGE_GATE_PX

        +

        IMAGE_GATE_PER_GAP
        *
        (
            frame_gap
            -
            1
        )

    )


    base_pitch_gate = (

        BASE_PITCH_GATE_M

        +

        PITCH_GATE_PER_GAP
        *
        (
            frame_gap
            -
            1
        )

    )


    options = []


    for candidate in (
        candidates
    ):

        is_dual = (
            candidate[
                "type"
            ]
            ==
            "dual"
        )


        if (

            not is_dual

            and

            candidate[
                "confidence"
            ]
            <
            SINGLE_TRACK_MIN_CONF

        ):

            continue


        if is_dual:

            image_gate = (
                base_image_gate
            )

            pitch_gate = (
                base_pitch_gate
            )

        else:

            image_gate = (

                base_image_gate

                *

                SINGLE_IMAGE_GATE_SCALE

            )


            pitch_gate = (

                base_pitch_gate

                *

                SINGLE_PITCH_GATE_SCALE

            )


        image_position = np.array(

            [
                candidate[
                    "image_x"
                ],

                candidate[
                    "image_y"
                ],
            ],

            dtype=np.float32,

        )


        pitch_position = np.array(

            [
                candidate[
                    "pitch_x_m"
                ],

                candidate[
                    "pitch_y_m"
                ],
            ],

            dtype=np.float32,

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


        if (
            speed
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


        image_cost = (

            image_residual

            /

            max(
                image_gate,
                1.0,
            )

        )


        pitch_cost = (

            pitch_residual

            /

            max(
                pitch_gate,
                0.5,
            )

        )


        confidence_cost = (

            1.0

            -

            candidate[
                "confidence"
            ]

        )


        support_cost = (

            0.0

            if is_dual

            else 1.0

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

                pitch_residual,

            )
        )


    if not options:

        return (
            None,
            None,
            None,
            None,
            None,
        )


    options.sort(
        key=lambda x:
            x[0]
    )


    (
        score,
        candidate,
        speed,
        image_residual,
        pitch_residual,
    ) = options[0]


    return (

        candidate,

        float(
            score
        ),

        float(
            speed
        ),

        float(
            image_residual
        ),

        float(
            pitch_residual
        ),

    )


# ============================================================
# Reacquisition
# ============================================================

def reacquire_eligible(
    candidate,
):

    if (
        candidate[
            "type"
        ]
        ==
        "dual"
    ):

        return True


    return (

        candidate[
            "confidence"
        ]

        >=

        SINGLE_REACQUIRE_MIN_CONF

    )


def choose_seed(
    candidates,
):

    eligible = [

        candidate

        for candidate
        in candidates

        if reacquire_eligible(
            candidate
        )

    ]


    if not eligible:

        return None


    eligible.sort(

        key=lambda candidate:
        (

            0

            if
            candidate[
                "type"
            ]
            ==
            "dual"

            else
            1,

            -candidate[
                "confidence"
            ],

        )

    )


    return eligible[0]


def confirm_seed(
    pending,
    candidates,
    frame_idx,
    fps,
):

    if pending is None:

        return None


    frame_gap = (

        frame_idx

        -

        pending[
            "frame"
        ]

    )


    if (

        frame_gap
        <=
        0

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
            pending[
                "image_x"
            ],

            pending[
                "image_y"
            ],
        ],

        dtype=np.float32,

    )


    old_pitch = np.array(

        [
            pending[
                "pitch_x_m"
            ],

            pending[
                "pitch_y_m"
            ],
        ],

        dtype=np.float32,

    )


    options = []


    for candidate in (
        candidates
    ):

        if not reacquire_eligible(
            candidate
        ):

            continue


        image = np.array(

            [
                candidate[
                    "image_x"
                ],

                candidate[
                    "image_y"
                ],
            ],

            dtype=np.float32,

        )


        pitch = np.array(

            [
                candidate[
                    "pitch_x_m"
                ],

                candidate[
                    "pitch_y_m"
                ],
            ],

            dtype=np.float32,

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

            if
            candidate[
                "type"
            ]
            ==
            "dual"

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
                candidate[
                    "confidence"
                ]
            )

            +

            0.10
            *
            support_penalty

        )


        options.append(
            (
                score,
                candidate,
            )
        )


    if not options:

        return None


    options.sort(
        key=lambda x:
            x[0]
    )


    return (
        options[0][1]
    )


# ============================================================
# PASS A
#
# Ensemble detection + temporal association
# ============================================================

def run_ensemble(
    video_path,
    raw_output_csv,
):

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "Invalid video FPS."
        )


    last_H = None

    fallback_streak = 0


    history = deque(
        maxlen=2
    )


    pending = None

    segment_id = -1

    rows = []


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

    print(
        "=" * 58
    )

    print(
        "PASS A — BALL ENSEMBLE TRACKING"
    )

    print(
        "=" * 58
    )


    print(
        "Frames:",
        total_frames,
    )


    print(
        "FPS:",
        fps,
    )


    while True:

        ret, frame = (
            cap.read()
        )


        if not ret:

            break


        (
            H,
            last_H,
            fallback_streak,
            homography_fallback,
            valid_kp,
            inliers,
            mean_error,
            median_error,
        ) = estimate_homography(

            frame,

            last_H,

            fallback_streak,

        )


        polygon = (
            get_pitch_polygon(
                H
            )
        )


        old_raw = detect_model(

            old_ball_model,

            frame,

            OLD_BALL_CLASS,

            OLD_CONF,

            "old",

        )


        forza_raw = detect_model(

            forza_model,

            frame,

            FORZA_BALL_CLASS,

            FORZA_CONF,

            "forza",

        )


        old_raw_total += len(
            old_raw
        )


        forza_raw_total += len(
            forza_raw
        )


        ensemble = (
            build_ensemble_candidates(

                old_raw,

                forza_raw,

            )
        )


        dual_count = sum(

            candidate[
                "type"
            ]
            ==
            "dual"

            for candidate
            in ensemble

        )


        single_count = (

            len(
                ensemble
            )

            -

            dual_count

        )


        dual_candidate_total += (
            dual_count
        )


        single_candidate_total += (
            single_count
        )


        (
            candidates,
            rejected,
        ) = pitch_filter_candidates(

            ensemble,

            H,

            polygon,

        )


        pitch_rejected_total += len(
            rejected
        )


        ball = None

        mode = "missing"

        score = None

        speed = None

        image_residual = None

        pitch_residual = None


        active = False


        if len(
            history
        ) > 0:

            gap = (

                frame_idx

                -

                history[-1][
                    "frame"
                ]

            )


            active = (
                gap
                <=
                MAX_ASSOCIATION_GAP
            )


        if active:

            (
                ball,
                score,
                speed,
                image_residual,
                pitch_residual,
            ) = associate_active(

                candidates,

                history,

                frame_idx,

                fps,

            )


            if ball is not None:

                mode = (
                    "tracked"
                )

                pending = None


            else:

                mode = (
                    "short_gap_missing"
                )


        else:

            confirmed = (
                confirm_seed(

                    pending,

                    candidates,

                    frame_idx,

                    fps,

                )
            )


            if confirmed is not None:

                ball = (
                    confirmed
                )

                mode = (
                    "reacquired"
                )

                segment_id += 1

                history.clear()

                pending = None

                reacquisitions += 1


            else:

                seed = (
                    choose_seed(
                        candidates
                    )
                )


                if seed is not None:

                    pending = {

                        **seed,

                        "frame":
                            frame_idx,

                    }


                    mode = (
                        "pending"
                    )


                    pending_frames += 1


                else:

                    pending = None

                    mode = (
                        "waiting"
                    )


        accepted = (
            ball is not None
        )


        if accepted:

            if (
                segment_id
                <
                0
            ):

                segment_id = 0


            accepted_frames += 1


            if (
                ball[
                    "type"
                ]
                ==
                "dual"
            ):

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
                        ball[
                            "image_x"
                        ],

                    "image_y":
                        ball[
                            "image_y"
                        ],

                    "pitch_x_m":
                        ball[
                            "pitch_x_m"
                        ],

                    "pitch_y_m":
                        ball[
                            "pitch_y_m"
                        ],

                }
            )


        else:

            missing_frames += 1


        rows.append(
            {

                "frame":
                    frame_idx,

                "time_sec":
                    frame_idx
                    /
                    fps,

                "old_raw_candidates":
                    len(
                        old_raw
                    ),

                "forza_raw_candidates":
                    len(
                        forza_raw
                    ),

                "dual_candidates":
                    dual_count,

                "single_candidates":
                    single_count,

                "pitch_valid_candidates":
                    len(
                        candidates
                    ),

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
                        ball[
                            "type"
                        ]
                        if accepted
                        else None
                    ),

                "confidence":
                    (
                        ball[
                            "confidence"
                        ]
                        if accepted
                        else None
                    ),

                "old_conf":
                    (
                        ball[
                            "old_conf"
                        ]
                        if accepted
                        else None
                    ),

                "forza_conf":
                    (
                        ball[
                            "forza_conf"
                        ]
                        if accepted
                        else None
                    ),

                "image_x":
                    (
                        ball[
                            "image_x"
                        ]
                        if accepted
                        else None
                    ),

                "image_y":
                    (
                        ball[
                            "image_y"
                        ]
                        if accepted
                        else None
                    ),

                "pitch_x_m":
                    (
                        ball[
                            "pitch_x_m"
                        ]
                        if accepted
                        else None
                    ),

                "pitch_y_m":
                    (
                        ball[
                            "pitch_y_m"
                        ]
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
                    homography_fallback,

            }
        )


        if (
            frame_idx
            %
            30
            ==
            0
        ):

            print(

                f"frame={frame_idx:04d} | "

                f"old={len(old_raw)} | "

                f"forza={len(forza_raw)} | "

                f"dual={dual_count} | "

                f"single={single_count} | "

                f"accepted={accepted} | "

                f"mode={mode}"

            )


        frame_idx += 1


    cap.release()


    if not rows:

        raise RuntimeError(
            "Ball ensemble produced no rows."
        )


    ensure_parent(
        raw_output_csv
    )


    with open(

        raw_output_csv,

        "w",

        newline="",

        encoding="utf-8",

    ) as f:

        fieldnames = list(
            rows[0].keys()
        )


        writer = csv.DictWriter(

            f,

            fieldnames=fieldnames,

        )


        writer.writeheader()


        writer.writerows(
            rows
        )


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

        dtype=np.float32,

    )


    print("")

    print(
        "ENSEMBLE SUMMARY"
    )


    print(
        "Frames:",
        frame_idx,
    )


    print(
        "Old detector candidates:",
        old_raw_total,
    )


    print(
        "Forza detector candidates:",
        forza_raw_total,
    )


    print(
        "Dual-model candidates:",
        dual_candidate_total,
    )


    print(
        "Single-model candidates:",
        single_candidate_total,
    )


    print(
        "Pitch rejected:",
        pitch_rejected_total,
    )


    print(
        "Accepted frames:",
        accepted_frames,
    )


    print(
        "Tracking coverage:",
        f"{coverage:.1f}%",
    )


    print(
        "Missing frames:",
        missing_frames,
    )


    print(
        "Dual accepted:",
        dual_accepted,
    )


    print(
        "Single accepted:",
        single_accepted,
    )


    print(
        "Dual share:",
        f"{dual_share:.1f}%",
    )


    print(
        "Pending frames:",
        pending_frames,
    )


    print(
        "Confirmed reacquisitions:",
        reacquisitions,
    )


    print(
        "Trajectory segments:",
        (
            segment_id + 1
            if segment_id >= 0
            else 0
        ),
    )


    if len(
        speeds
    ) > 0:

        print(
            "Speed median:",
            f"{np.median(speeds):.2f} m/s",
        )


        print(
            "Speed P95:",
            f"{np.percentile(speeds, 95):.2f} m/s",
        )


        print(
            "Speed P99:",
            f"{np.percentile(speeds, 99):.2f} m/s",
        )


        print(
            "Speed max:",
            f"{np.max(speeds):.2f} m/s",
        )


    print(
        "Raw ensemble CSV:",
        raw_output_csv,
    )


    return fps


# ============================================================
# PASS B
#
# Trusted filtering + interpolation
# ============================================================

def build_trusted_ball(
    raw_output_csv,
    output_csv,
    fps,
):

    df = pd.read_csv(
        raw_output_csv
    )


    df = (

        df

        .sort_values(
            "frame"
        )

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # Normalize
    # ========================================================

    df[
        "accepted_bool"
    ] = (

        df[
            "accepted"
        ]

        .astype(
            str
        )

        .str.lower()

        .isin(
            [
                "true",
                "1",
            ]
        )

    )


    for col in [

        "segment_id",

        "confidence",

        "image_residual_px",

        "pitch_residual_m",

        "estimated_speed_mps",

    ]:

        df[
            col
        ] = pd.to_numeric(

            df[
                col
            ],

            errors="coerce",

        )


    # ========================================================
    # Trusted columns
    # ========================================================

    df[
        "ball_trusted_direct"
    ] = False


    df[
        "ball_interpolated"
    ] = False


    df[
        "ball_usable"
    ] = False


    df[
        "ball_quality"
    ] = "missing"


    df[
        "trust_reason"
    ] = "missing"


    df[
        "trusted_segment_id"
    ] = np.nan


    df[
        "image_x_trusted"
    ] = np.nan


    df[
        "image_y_trusted"
    ] = np.nan


    df[
        "pitch_x_trusted_m"
    ] = np.nan


    df[
        "pitch_y_trusted_m"
    ] = np.nan


    # ========================================================
    # Direct observation validation
    # ========================================================

    dual_trusted_count = 0

    single_trusted_count = 0

    single_rejected_count = 0


    for i in range(
        len(
            df
        )
    ):

        if not df.loc[
            i,
            "accepted_bool",
        ]:

            continue


        candidate_type = str(

            df.loc[
                i,
                "candidate_type",
            ]

        )


        mode = str(

            df.loc[
                i,
                "mode",
            ]

        )


        confidence = df.loc[
            i,
            "confidence",
        ]


        # ----------------------------------------------------
        # DUAL
        # ----------------------------------------------------

        if (
            candidate_type
            ==
            "dual"
        ):

            df.loc[
                i,
                "ball_trusted_direct",
            ] = True


            df.loc[
                i,
                "ball_quality",
            ] = "trusted_dual"


            df.loc[
                i,
                "trust_reason",
            ] = (
                "cross_model_agreement"
            )


            dual_trusted_count += 1

            continue


        # ----------------------------------------------------
        # SINGLE type
        # ----------------------------------------------------

        if candidate_type not in [

            "old_single",

            "forza_single",

        ]:

            df.loc[
                i,
                "ball_quality",
            ] = "rejected"


            df.loc[
                i,
                "trust_reason",
            ] = (
                "unknown_candidate_type"
            )


            continue


        # ----------------------------------------------------
        # Reacquired single
        # ----------------------------------------------------

        if (
            mode
            ==
            "reacquired"
        ):

            if (

                pd.notna(
                    confidence
                )

                and

                confidence
                >=
                TRUST_SINGLE_REACQUIRE_MIN_CONF

            ):

                df.loc[
                    i,
                    "ball_trusted_direct",
                ] = True


                df.loc[
                    i,
                    "ball_quality",
                ] = "trusted_single"


                df.loc[
                    i,
                    "trust_reason",
                ] = (
                    "single_confirmed_reacquisition"
                )


                single_trusted_count += 1


            else:

                df.loc[
                    i,
                    "ball_quality",
                ] = "rejected"


                df.loc[
                    i,
                    "trust_reason",
                ] = (
                    "single_reacquire_low_conf"
                )


                single_rejected_count += 1


            continue


        # ----------------------------------------------------
        # Normal tracked single
        # ----------------------------------------------------

        image_residual = df.loc[
            i,
            "image_residual_px",
        ]


        pitch_residual = df.loc[
            i,
            "pitch_residual_m",
        ]


        speed = df.loc[
            i,
            "estimated_speed_mps",
        ]


        confidence_ok = (

            pd.notna(
                confidence
            )

            and

            confidence
            >=
            TRUST_SINGLE_MIN_CONF

        )


        image_ok = (

            pd.notna(
                image_residual
            )

            and

            image_residual
            <=
            TRUST_SINGLE_MAX_IMAGE_RESIDUAL_PX

        )


        pitch_ok = (

            pd.notna(
                pitch_residual
            )

            and

            pitch_residual
            <=
            TRUST_SINGLE_MAX_PITCH_RESIDUAL_M

        )


        speed_ok = (

            pd.notna(
                speed
            )

            and

            speed
            <=
            TRUST_SINGLE_MAX_SPEED_MPS

        )


        if (

            confidence_ok

            and

            image_ok

            and

            pitch_ok

            and

            speed_ok

        ):

            df.loc[
                i,
                "ball_trusted_direct",
            ] = True


            df.loc[
                i,
                "ball_quality",
            ] = "trusted_single"


            df.loc[
                i,
                "trust_reason",
            ] = (
                "single_temporally_supported"
            )


            single_trusted_count += 1


        else:

            df.loc[
                i,
                "ball_quality",
            ] = "rejected"


            df.loc[
                i,
                "trust_reason",
            ] = (
                "single_failed_strict_validation"
            )


            single_rejected_count += 1


    # ========================================================
    # Copy trusted observations
    # ========================================================

    mask = (
        df[
            "ball_trusted_direct"
        ]
    )


    df.loc[
        mask,
        "ball_usable",
    ] = True


    df.loc[
        mask,
        "trusted_segment_id",
    ] = df.loc[
        mask,
        "segment_id",
    ]


    df.loc[
        mask,
        "image_x_trusted",
    ] = df.loc[
        mask,
        "image_x",
    ]


    df.loc[
        mask,
        "image_y_trusted",
    ] = df.loc[
        mask,
        "image_y",
    ]


    df.loc[
        mask,
        "pitch_x_trusted_m",
    ] = df.loc[
        mask,
        "pitch_x_m",
    ]


    df.loc[
        mask,
        "pitch_y_trusted_m",
    ] = df.loc[
        mask,
        "pitch_y_m",
    ]


    # ========================================================
    # Short-gap interpolation
    # ========================================================

    interpolated_count = 0

    n = len(
        df
    )

    i = 0


    while i < n:

        if df.loc[
            i,
            "ball_trusted_direct",
        ]:

            i += 1

            continue


        gap_start = i


        while (

            i < n

            and

            not df.loc[
                i,
                "ball_trusted_direct",
            ]

        ):

            i += 1


        gap_end = (
            i - 1
        )


        gap_length = (

            gap_end

            -

            gap_start

            +

            1

        )


        previous_index = (
            gap_start
            -
            1
        )


        next_index = i


        if (
            gap_length
            >
            MAX_INTERPOLATION_GAP
        ):

            continue


        if (

            previous_index
            <
            0

            or

            next_index
            >=
            n

        ):

            continue


        if not (

            df.loc[
                previous_index,
                "ball_trusted_direct",
            ]

            and

            df.loc[
                next_index,
                "ball_trusted_direct",
            ]

        ):

            continue


        previous_segment = (
            df.loc[
                previous_index,
                "segment_id",
            ]
        )


        next_segment = (
            df.loc[
                next_index,
                "segment_id",
            ]
        )


        if (

            pd.isna(
                previous_segment
            )

            or

            pd.isna(
                next_segment
            )

            or

            previous_segment
            !=
            next_segment

        ):

            continue


        frame_prev = df.loc[
            previous_index,
            "frame",
        ]


        frame_next = df.loc[
            next_index,
            "frame",
        ]


        dt = (

            frame_next

            -

            frame_prev

        ) / fps


        if dt <= 0:

            continue


        x_prev = df.loc[
            previous_index,
            "pitch_x_m",
        ]


        y_prev = df.loc[
            previous_index,
            "pitch_y_m",
        ]


        x_next = df.loc[
            next_index,
            "pitch_x_m",
        ]


        y_next = df.loc[
            next_index,
            "pitch_y_m",
        ]


        distance = float(

            np.hypot(

                x_next
                -
                x_prev,

                y_next
                -
                y_prev,

            )

        )


        endpoint_speed = (
            distance
            /
            dt
        )


        if (
            endpoint_speed
            >
            MAX_INTERPOLATION_SPEED_MPS
        ):

            continue


        ix_prev = df.loc[
            previous_index,
            "image_x",
        ]


        iy_prev = df.loc[
            previous_index,
            "image_y",
        ]


        ix_next = df.loc[
            next_index,
            "image_x",
        ]


        iy_next = df.loc[
            next_index,
            "image_y",
        ]


        total_steps = (
            gap_length
            +
            1
        )


        for offset in range(

            1,

            gap_length
            +
            1,

        ):

            row_index = (

                previous_index

                +

                offset

            )


            alpha = (

                offset

                /

                total_steps

            )


            df.loc[
                row_index,
                "pitch_x_trusted_m",
            ] = (

                x_prev

                +

                alpha
                *
                (
                    x_next
                    -
                    x_prev
                )

            )


            df.loc[
                row_index,
                "pitch_y_trusted_m",
            ] = (

                y_prev

                +

                alpha
                *
                (
                    y_next
                    -
                    y_prev
                )

            )


            df.loc[
                row_index,
                "image_x_trusted",
            ] = (

                ix_prev

                +

                alpha
                *
                (
                    ix_next
                    -
                    ix_prev
                )

            )


            df.loc[
                row_index,
                "image_y_trusted",
            ] = (

                iy_prev

                +

                alpha
                *
                (
                    iy_next
                    -
                    iy_prev
                )

            )


            df.loc[
                row_index,
                "trusted_segment_id",
            ] = (
                previous_segment
            )


            df.loc[
                row_index,
                "ball_usable",
            ] = True


            df.loc[
                row_index,
                "ball_interpolated",
            ] = True


            df.loc[
                row_index,
                "ball_quality",
            ] = "interpolated"


            df.loc[
                row_index,
                "trust_reason",
            ] = (
                "short_gap_interpolation"
            )


            interpolated_count += 1


    # ========================================================
    # Trusted speed
    # ========================================================

    df[
        "trusted_speed_mps"
    ] = np.nan


    trusted_df = df[
        df[
            "ball_usable"
        ]
    ].copy()


    for (
        _,
        segment,
    ) in trusted_df.groupby(

        "trusted_segment_id",

        dropna=True,

    ):

        segment = (
            segment
            .sort_values(
                "frame"
            )
        )


        indices = (
            segment
            .index
            .tolist()
        )


        previous_index = None


        for current_index in (
            indices
        ):

            if previous_index is None:

                previous_index = (
                    current_index
                )

                continue


            frame_gap = (

                df.loc[
                    current_index,
                    "frame",
                ]

                -

                df.loc[
                    previous_index,
                    "frame",
                ]

            )


            if (

                frame_gap
                <=
                0

                or

                frame_gap
                >
                5

            ):

                previous_index = (
                    current_index
                )

                continue


            dt = (
                frame_gap
                /
                fps
            )


            distance = float(

                np.hypot(

                    df.loc[
                        current_index,
                        "pitch_x_trusted_m",
                    ]

                    -

                    df.loc[
                        previous_index,
                        "pitch_x_trusted_m",
                    ],

                    df.loc[
                        current_index,
                        "pitch_y_trusted_m",
                    ]

                    -

                    df.loc[
                        previous_index,
                        "pitch_y_trusted_m",
                    ],

                )

            )


            df.loc[
                current_index,
                "trusted_speed_mps",
            ] = (
                distance
                /
                dt
            )


            previous_index = (
                current_index
            )


    # ========================================================
    # Remaining gaps
    # ========================================================

    gaps = []

    gap = 0


    for usable in (
        df[
            "ball_usable"
        ]
    ):

        if not usable:

            gap += 1


        else:

            if gap > 0:

                gaps.append(
                    gap
                )

                gap = 0


    if gap > 0:

        gaps.append(
            gap
        )


    longest_gap = (

        max(
            gaps
        )

        if gaps

        else 0

    )


    # ========================================================
    # Save
    # ========================================================

    ensure_parent(
        output_csv
    )


    df.to_csv(

        output_csv,

        index=False,

    )


    # ========================================================
    # Summary
    # ========================================================

    usable_count = int(

        df[
            "ball_usable"
        ].sum()

    )


    usable_coverage = (

        usable_count

        /

        len(
            df
        )

        *

        100.0

    )


    speeds = (

        df[
            "trusted_speed_mps"
        ]

        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        .dropna()

    )


    print("")

    print(
        "=" * 58
    )

    print(
        "PASS B — TRUSTED BALL CLEANING"
    )

    print(
        "=" * 58
    )


    print(
        "Frames:",
        len(
            df
        ),
    )


    print(
        "Trusted DUAL:",
        dual_trusted_count,
    )


    print(
        "Trusted SINGLE:",
        single_trusted_count,
    )


    print(
        "Rejected SINGLE:",
        single_rejected_count,
    )


    print(
        "Interpolated:",
        interpolated_count,
    )


    print(
        "Final usable frames:",
        usable_count,
    )


    print(
        "Final usable coverage:",
        f"{usable_coverage:.1f}%",
    )


    print(
        "Missing / unknown:",
        int(
            (
                ~df[
                    "ball_usable"
                ]
            ).sum()
        ),
    )


    print(
        "Longest remaining gap:",
        longest_gap,
        "frames",
    )


    if len(
        speeds
    ) > 0:

        print(
            "Speed median:",
            f"{speeds.median():.2f} m/s",
        )


        print(
            "Speed P95:",
            f"{speeds.quantile(0.95):.2f} m/s",
        )


        print(
            "Speed P99:",
            f"{speeds.quantile(0.99):.2f} m/s",
        )


        print(
            "Speed max:",
            f"{speeds.max():.2f} m/s",
        )


    print("")


    print(
        df[
            "ball_quality"
        ]
        .value_counts()
        .to_string()
    )


    print("")


    print(
        "Trusted CSV:",
        output_csv,
    )


    return df


# ============================================================
# Trusted QA
# ============================================================

def render_trusted_qa(
    video_path,
    df,
    output_video,
):

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
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


    fps_video = cap.get(
        cv2.CAP_PROP_FPS
    )


    writer = cv2.VideoWriter(

        str(
            output_video
        ),

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps_video,

        (
            width,
            height,
        ),

    )


    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Cannot create QA video: "
            f"{output_video}"
        )


    frame_idx = 0


    while True:

        ret, frame = (
            cap.read()
        )


        if not ret:

            break


        if (
            frame_idx
            >=
            len(
                df
            )
        ):

            break


        row = df.iloc[
            frame_idx
        ]


        quality = row[
            "ball_quality"
        ]


        # ----------------------------------------------------
        # Rejected accepted candidate
        # ----------------------------------------------------

        if (

            quality
            ==
            "rejected"

            and

            row[
                "accepted_bool"
            ]

            and

            pd.notna(
                row[
                    "image_x"
                ]
            )

            and

            pd.notna(
                row[
                    "image_y"
                ]
            )

        ):

            x = int(
                row[
                    "image_x"
                ]
            )


            y = int(
                row[
                    "image_y"
                ]
            )


            cv2.circle(

                frame,

                (
                    x,
                    y,
                ),

                10,

                (
                    0,
                    0,
                    255,
                ),

                3,

            )


            cv2.putText(

                frame,

                "REJECT",

                (
                    x + 12,
                    y - 8,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (
                    0,
                    0,
                    255,
                ),

                2,

                cv2.LINE_AA,

            )


        # ----------------------------------------------------
        # Trusted / interpolated
        # ----------------------------------------------------

        if row[
            "ball_usable"
        ]:

            x = int(
                row[
                    "image_x_trusted"
                ]
            )


            y = int(
                row[
                    "image_y_trusted"
                ]
            )


            if (
                quality
                ==
                "trusted_dual"
            ):

                color = (
                    0,
                    255,
                    0,
                )

                label = (
                    "BALL DUAL"
                )


            elif (
                quality
                ==
                "trusted_single"
            ):

                color = (
                    0,
                    165,
                    255,
                )

                label = (
                    "BALL SINGLE"
                )


            else:

                color = (
                    255,
                    255,
                    0,
                )

                label = (
                    "BALL INTERP"
                )


            cv2.circle(

                frame,

                (
                    x,
                    y,
                ),

                8,

                color,

                -1,

            )


            cv2.putText(

                frame,

                label,

                (
                    x + 10,
                    y - 8,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                color,

                2,

                cv2.LINE_AA,

            )


        cv2.putText(

            frame,

            (
                f"Trusted Ball | "
                f"{quality}"
            ),

            (
                20,
                35,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                255,
                255,
                255,
            ),

            2,

            cv2.LINE_AA,

        )


        writer.write(
            frame
        )


        frame_idx += 1


    cap.release()

    writer.release()


    print(
        "QA Video:",
        output_video,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Canonical two-pass ball tracking: "

            "cross-model ensemble tracking followed by "

            "trusted filtering/interpolation."

        )

    )


    parser.add_argument(
        "--video",
        required=True,
    )


    parser.add_argument(

        "--old-ball-model",

        default=(
            "models/"
            "football-ball-detection.pt"
        ),

    )


    parser.add_argument(

        "--forza-model",

        default=(
            "models/"
            "yolov8m_forzasys_soccer.pt"
        ),

    )


    parser.add_argument(

        "--pitch-model",

        default=(
            "models/"
            "football-pitch-detection.pt"
        ),

    )


    parser.add_argument(
        "--raw-output-csv",
        required=True,
    )


    parser.add_argument(
        "--output-csv",
        required=True,
    )


    parser.add_argument(
        "--output-video",
        default=None,
    )


    parser.add_argument(
        "--device",
        default="auto",
    )


    parser.add_argument(

        "--reuse-raw",

        action="store_true",

        help=(
            "Skip detector pass and reuse "
            "--raw-output-csv."
        ),

    )


    parser.add_argument(
        "--skip-qa",
        action="store_true",
    )


    args = (
        parser.parse_args()
    )


    # ========================================================
    # Input validation
    # ========================================================

    require_file(
        args.video,
        "video",
    )


    if args.reuse_raw:

        require_file(

            args.raw_output_csv,

            "raw ensemble CSV",

        )


    else:

        for (
            path,
            label,
        ) in [

            (
                args.old_ball_model,
                "old ball model",
            ),

            (
                args.forza_model,
                "Forza ball model",
            ),

            (
                args.pitch_model,
                "pitch model",
            ),

        ]:

            require_file(
                path,
                label,
            )


    ensure_parent(
        args.raw_output_csv
    )


    ensure_parent(
        args.output_csv
    )


    if (
        args.output_video
        is not None
    ):

        ensure_parent(
            args.output_video
        )


    # ========================================================
    # Runtime
    # ========================================================

    global DEVICE

    global old_ball_model

    global forza_model

    global pitch_model


    DEVICE = resolve_device(
        args.device
    )


    if not args.reuse_raw:

        old_ball_model = YOLO(
            args.old_ball_model
        )


        forza_model = YOLO(
            args.forza_model
        )


        pitch_model = YOLO(
            args.pitch_model
        )


    print("")

    print(
        "=" * 58
    )

    print(
        "BALL TRACKING CLI"
    )

    print(
        "=" * 58
    )


    print(
        "Video:       ",
        args.video,
    )


    print(
        "Old model:   ",
        args.old_ball_model,
    )


    print(
        "Forza model: ",
        args.forza_model,
    )


    print(
        "Pitch model: ",
        args.pitch_model,
    )


    print(
        "Device:      ",
        DEVICE,
    )


    print(
        "Raw CSV:     ",
        args.raw_output_csv,
    )


    print(
        "Trusted CSV: ",
        args.output_csv,
    )


    print(
        "QA video:    ",
        args.output_video,
    )


    # ========================================================
    # PASS A
    # ========================================================

    if args.reuse_raw:

        print(
            "Reusing raw ensemble CSV; "
            "detector pass skipped."
        )


        fps = get_video_fps(
            args.video
        )


    else:

        fps = run_ensemble(

            args.video,

            args.raw_output_csv,

        )


    # ========================================================
    # PASS B
    # ========================================================

    trusted_df = (
        build_trusted_ball(

            args.raw_output_csv,

            args.output_csv,

            fps,

        )
    )


    # ========================================================
    # QA
    # ========================================================

    if not args.skip_qa:

        if (
            args.output_video
            is None
        ):

            raise ValueError(

                "--output-video is required "
                "unless --skip-qa is used."

            )


        render_trusted_qa(

            args.video,

            trusted_df,

            args.output_video,

        )


    print("")

    print(
        "=" * 58
    )

    print(
        "DONE"
    )

    print(
        "=" * 58
    )


    print(
        "Raw CSV:    ",
        args.raw_output_csv,
    )


    print(
        "Trusted CSV:",
        args.output_csv,
    )


    if not args.skip_qa:

        print(
            "QA Video:   ",
            args.output_video,
        )


if __name__ == "__main__":

    main()
