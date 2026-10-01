#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# ============================================================
# Validated AirTouch V3 parameters
# ============================================================

GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2

PLAYER_CONF = 0.25
PLAYER_IMGSZ = 1280

MIN_DOMINANT_TEAM_SHARE = 0.70
MAX_TRACK_BOX_MATCH_PX = 55.0

BOX_X_MARGIN_RATIO = 0.18
BOX_Y_MARGIN_RATIO = 0.10

# normalized bbox Y:
# 0.0 = top
# 1.0 = feet
AIR_ZONE_Y_MAX = 0.58

VELOCITY_OFFSET = 2
MIN_VECTOR_LENGTH_PX = 4.0

STRONG_DIRECTION_CHANGE_DEG = 25.0

COMBINED_DIRECTION_CHANGE_DEG = 18.0
COMBINED_SPEED_CHANGE_RATIO = 0.35

VERY_STRONG_MIN_SPEED_MPS = 18.0
VERY_STRONG_DIRECTION_CHANGE_DEG = 60.0

RECEPTION_LOOKAHEAD_FRAMES = 7

MAX_EVENT_FRAME_GAP = 2

ALLOWED_BALL_STATES = {
    "InTransit",
    "AerialContest",
}


# ============================================================
# Validated V3 regression frames
# ============================================================

NEGATIVE_FRAME = 203
NEGATIVE_TEAM = "A"
NEGATIVE_TRACK_ID = 6

POSITIVE_FRAME = 281
POSITIVE_TEAM = "B"
POSITIVE_TRACK_ID = 10


# ============================================================
# Validated AirTouch V4 parameters
# ============================================================

ARBITRATION_FRAME_GAP = 1

CLEAR_WINNER_DISTANCE_RATIO = 0.55
CLEAR_WINNER_DISTANCE_MARGIN_NORM = 0.18


# ============================================================
# Validated V4 regression cases
# ============================================================

CONTEST_FRAME = 172

RECEPTION_FRAME = 203
RECEPTION_TEAM = "A"
RECEPTION_TRACK_ID = 6

HEADER_FRAME = 281
HEADER_TEAM = "B"
HEADER_TRACK_ID = 10


# ============================================================
# General helpers
# ============================================================

def require_file(
    path,
    label,
):

    path = Path(
        path
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: "
            f"{path}"
        )

    if (
        path.stat().st_size
        ==
        0
    ):

        raise RuntimeError(
            f"{label} is empty: "
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


def resolve_device(
    value,
):

    if (
        value
        !=
        "auto"
    ):

        return value


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


    return "cpu"


def as_bool(
    value,
):

    if isinstance(
        value,
        bool,
    ):

        return value


    if pd.isna(
        value
    ):

        return False


    return (
        str(
            value
        )
        .strip()
        .lower()

        in

        {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def same_player_id(
    a,
    b,
):

    if (
        pd.isna(a)
        or
        pd.isna(b)
    ):

        return False


    return (
        abs(
            float(a)
            -
            float(b)
        )
        <
        0.1
    )


def participant_label(
    team,
    track_id,
):

    return (
        f"{team}"
        f"#{int(float(track_id))}"
    )


def parse_event_ids(
    value,
):

    if pd.isna(
        value
    ):

        return []


    ids = []


    for part in (
        str(
            value
        )
        .split(";")
    ):

        part = (
            part.strip()
        )


        if not part:

            continue


        ids.append(
            int(
                float(
                    part
                )
            )
        )


    return ids


# ============================================================
# Input normalization
# ============================================================

def prepare_inputs(
    player_csv,
    possession_csv,
):

    tracks = pd.read_csv(
        player_csv
    )


    possession = pd.read_csv(
        possession_csv
    )


    required_track = [
        "frame",
        "track_id",
        "image_x",
        "image_y",
        "team",
    ]


    missing = [
        column

        for column
        in required_track

        if column
        not in
        tracks.columns
    ]


    if missing:

        raise ValueError(
            "Player CSV missing required columns: "
            +
            ", ".join(
                missing
            )
        )


    required_possession = [
        "frame",

        "ball_image_x",
        "ball_image_y",
        "ball_speed_mps",

        "ball_state_v4",
        "possession_v4",

        "controller_team_v4",
        "controller_id_v4",

        "reception_candidate_team",
        "reception_candidate_id",
        "reception_candidate_count",
    ]


    missing = [
        column

        for column
        in required_possession

        if column
        not in
        possession.columns
    ]


    if missing:

        raise ValueError(
            "Possession seed CSV missing required columns: "
            +
            ", ".join(
                missing
            )
        )


    # ========================================================
    # Normalize player tracks
    # ========================================================

    for column in [
        "frame",
        "track_id",
        "image_x",
        "image_y",
    ]:

        tracks[
            column
        ] = pd.to_numeric(
            tracks[
                column
            ],
            errors="coerce",
        )


    tracks = (
        tracks
        .dropna(
            subset=[
                "frame",
                "track_id",
                "image_x",
                "image_y",
            ]
        )
        .copy()
    )


    tracks[
        "frame"
    ] = (
        tracks[
            "frame"
        ]
        .astype(int)
    )


    tracks = (
        tracks[
            tracks[
                "team"
            ].isin(
                [
                    "A",
                    "B",
                ]
            )
        ]
        .copy()
    )


    # ========================================================
    # Stable team reconstruction
    # ========================================================

    stable_team_map = {}


    for (
        track_id,
        group,

    ) in tracks.groupby(
        "track_id"
    ):

        counts = (
            group[
                "team"
            ]
            .value_counts()
        )


        if len(
            counts
        ) == 0:

            continue


        dominant_team = (
            counts.index[
                0
            ]
        )


        dominant_share = (
            counts.iloc[
                0
            ]
            /
            len(
                group
            )
        )


        if (
            dominant_share
            >=
            MIN_DOMINANT_TEAM_SHARE
        ):

            stable_team_map[
                track_id
            ] = dominant_team


    tracks[
        "stable_team"
    ] = (
        tracks[
            "track_id"
        ]
        .map(
            stable_team_map
        )
    )


    tracks = (
        tracks[
            tracks[
                "stable_team"
            ].notna()
        ]
        .copy()
    )


    # Never relabel inconsistent rows.
    tracks = (
        tracks[
            tracks[
                "team"
            ]
            ==
            tracks[
                "stable_team"
            ]
        ]
        .copy()
    )


    tracks_by_frame = {

        int(frame):
            group.copy()

        for (
            frame,
            group,
        )
        in tracks.groupby(
            "frame"
        )
    }


    # ========================================================
    # Normalize possession seed
    # ========================================================

    numeric_columns = [
        "frame",
        "time_sec",

        "ball_image_x",
        "ball_image_y",
        "ball_speed_mps",

        "controller_id_v4",

        "reception_candidate_id",
        "reception_candidate_count",
    ]


    for column in (
        numeric_columns
    ):

        if (
            column
            in
            possession.columns
        ):

            possession[
                column
            ] = pd.to_numeric(
                possession[
                    column
                ],
                errors="coerce",
            )


    possession = (
        possession
        .dropna(
            subset=[
                "frame"
            ]
        )
        .copy()
    )


    possession[
        "frame"
    ] = (
        possession[
            "frame"
        ]
        .astype(int)
    )


    possession_by_frame = (
        possession
        .set_index(
            "frame"
        )
    )


    return (
        tracks,
        tracks_by_frame,
        possession,
        possession_by_frame,
    )


# ============================================================
# Possession / ball helpers
# ============================================================

def get_possession_row(
    possession_by_frame,
    frame,
):

    if (
        frame
        not in
        possession_by_frame.index
    ):

        return None


    row = (
        possession_by_frame.loc[
            frame
        ]
    )


    if isinstance(
        row,
        pd.DataFrame,
    ):

        row = (
            row.iloc[
                0
            ]
        )


    return row


def get_ball_point(
    possession_by_frame,
    frame,
):

    row = get_possession_row(
        possession_by_frame,
        frame,
    )


    if row is None:

        return None


    x = row.get(
        "ball_image_x",
        np.nan,
    )


    y = row.get(
        "ball_image_y",
        np.nan,
    )


    if (
        pd.isna(x)
        or
        pd.isna(y)
    ):

        return None


    return np.asarray(
        [
            float(x),
            float(y),
        ],
        dtype=np.float32,
    )


# ============================================================
# Trajectory deflection
# ============================================================

def calculate_deflection(
    possession_by_frame,
    frame,
):

    before = get_ball_point(
        possession_by_frame,
        frame
        -
        VELOCITY_OFFSET,
    )


    touch = get_ball_point(
        possession_by_frame,
        frame,
    )


    after = get_ball_point(
        possession_by_frame,
        frame
        +
        VELOCITY_OFFSET,
    )


    if any(
        point is None

        for point
        in [
            before,
            touch,
            after,
        ]
    ):

        return {
            "valid":
                False,

            "direction_change_deg":
                np.nan,

            "speed_change_ratio":
                np.nan,

            "before_length_px":
                np.nan,

            "after_length_px":
                np.nan,
        }


    vector_before = (
        touch
        -
        before
    )


    vector_after = (
        after
        -
        touch
    )


    length_before = float(
        np.linalg.norm(
            vector_before
        )
    )


    length_after = float(
        np.linalg.norm(
            vector_after
        )
    )


    if (
        length_before
        <
        MIN_VECTOR_LENGTH_PX

        or

        length_after
        <
        MIN_VECTOR_LENGTH_PX
    ):

        return {
            "valid":
                False,

            "direction_change_deg":
                np.nan,

            "speed_change_ratio":
                np.nan,

            "before_length_px":
                length_before,

            "after_length_px":
                length_after,
        }


    cosine = float(
        np.dot(
            vector_before,
            vector_after,
        )
        /
        (
            length_before
            *
            length_after
        )
    )


    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )


    direction_change = (
        math.degrees(
            math.acos(
                cosine
            )
        )
    )


    speed_change_ratio = (

        abs(
            length_after
            -
            length_before
        )

        /

        max(
            length_before,
            1e-6,
        )
    )


    return {
        "valid":
            True,

        "direction_change_deg":
            float(
                direction_change
            ),

        "speed_change_ratio":
            float(
                speed_change_ratio
            ),

        "before_length_px":
            length_before,

        "after_length_px":
            length_after,
    }


# ============================================================
# Strong deflection
# ============================================================

def is_strong_deflection(
    deflection,
):

    if not deflection[
        "valid"
    ]:

        return False


    direction = (
        deflection[
            "direction_change_deg"
        ]
    )


    speed_change = (
        deflection[
            "speed_change_ratio"
        ]
    )


    if (
        direction
        >=
        STRONG_DIRECTION_CHANGE_DEG
    ):

        return True


    if (

        direction
        >=
        COMBINED_DIRECTION_CHANGE_DEG

        and

        speed_change
        >=
        COMBINED_SPEED_CHANGE_RATIO

    ):

        return True


    return False


# ============================================================
# Very strong air touch
# ============================================================

def is_very_strong_air_touch(
    ball_speed,
    deflection,
):

    if not deflection[
        "valid"
    ]:

        return False


    if pd.isna(
        ball_speed
    ):

        return False


    return (

        float(
            ball_speed
        )
        >=
        VERY_STRONG_MIN_SPEED_MPS

        and

        deflection[
            "direction_change_deg"
        ]
        >=
        VERY_STRONG_DIRECTION_CHANGE_DEG
    )


# ============================================================
# Future same-player reception / control
# ============================================================

def followed_by_same_player_control(
    possession_by_frame,
    frame,
    team,
    track_id,
):

    for future_frame in range(

        frame + 1,

        frame
        +
        RECEPTION_LOOKAHEAD_FRAMES
        +
        1,
    ):

        row = get_possession_row(
            possession_by_frame,
            future_frame,
        )


        if row is None:

            continue


        candidate_team = row.get(
            "reception_candidate_team",
            None,
        )


        candidate_id = row.get(
            "reception_candidate_id",
            np.nan,
        )


        candidate_count = row.get(
            "reception_candidate_count",
            0,
        )


        if (

            str(
                candidate_team
            )
            ==
            str(
                team
            )

            and

            same_player_id(
                candidate_id,
                track_id,
            )

            and

            pd.notna(
                candidate_count
            )

            and

            float(
                candidate_count
            )
            >
            0
        ):

            return True


        controller_team = row.get(
            "controller_team_v4",
            None,
        )


        controller_id = row.get(
            "controller_id_v4",
            np.nan,
        )


        if (

            str(
                controller_team
            )
            ==
            str(
                team
            )

            and

            same_player_id(
                controller_id,
                track_id,
            )
        ):

            return True


    return False


# ============================================================
# Player detector
# ============================================================

def detect_player_boxes(
    model,
    frame,
    device,
):

    result = model(

        frame,

        imgsz=PLAYER_IMGSZ,

        conf=PLAYER_CONF,

        classes=[
            GOALKEEPER_CLASS,
            PLAYER_CLASS,
        ],

        device=device,

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


    confidences = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )


    for (
        box,
        confidence,

    ) in zip(
        xyxy,
        confidences,
    ):

        x1, y1, x2, y2 = (
            box
        )


        detections.append(
            {
                "box":
                    np.asarray(
                        [
                            x1,
                            y1,
                            x2,
                            y2,
                        ],
                        dtype=np.float32,
                    ),

                "bottom_x":
                    float(
                        (
                            x1
                            +
                            x2
                        )
                        /
                        2.0
                    ),

                "bottom_y":
                    float(
                        y2
                    ),

                "confidence":
                    float(
                        confidence
                    ),
            }
        )


    return detections


# ============================================================
# Track <-> detection matching
# ============================================================

def match_tracks_to_boxes(
    frame_tracks,
    detections,
):

    if (
        frame_tracks is None
        or
        len(
            frame_tracks
        ) == 0
        or
        len(
            detections
        ) == 0
    ):

        return []


    track_rows = list(
        frame_tracks.iterrows()
    )


    possible_matches = []


    for (
        track_local_index,
        (
            _,
            track,
        ),

    ) in enumerate(
        track_rows
    ):

        tx = float(
            track[
                "image_x"
            ]
        )


        ty = float(
            track[
                "image_y"
            ]
        )


        for (
            detection_index,
            detection,

        ) in enumerate(
            detections
        ):

            distance = float(
                np.hypot(

                    tx
                    -
                    detection[
                        "bottom_x"
                    ],

                    ty
                    -
                    detection[
                        "bottom_y"
                    ],
                )
            )


            if (
                distance
                <=
                MAX_TRACK_BOX_MATCH_PX
            ):

                possible_matches.append(
                    (
                        distance,
                        track_local_index,
                        detection_index,
                    )
                )


    possible_matches.sort(
        key=lambda item:
            item[
                0
            ]
    )


    used_tracks = set()
    used_detections = set()

    matches = []


    for (
        distance,
        track_local_index,
        detection_index,

    ) in possible_matches:

        if (
            track_local_index
            in
            used_tracks
        ):

            continue


        if (
            detection_index
            in
            used_detections
        ):

            continue


        used_tracks.add(
            track_local_index
        )


        used_detections.add(
            detection_index
        )


        (
            _,
            track,

        ) = track_rows[
            track_local_index
        ]


        detection = (
            detections[
                detection_index
            ]
        )


        matches.append(
            {
                "track_id":
                    float(
                        track[
                            "track_id"
                        ]
                    ),

                "team":
                    str(
                        track[
                            "stable_team"
                        ]
                    ),

                "box":
                    detection[
                        "box"
                    ],

                "box_confidence":
                    detection[
                        "confidence"
                    ],

                "match_distance_px":
                    float(
                        distance
                    ),
            }
        )


    return matches


# ============================================================
# Upper-body interaction zone
# ============================================================

def upper_body_metrics(
    ball_x,
    ball_y,
    box,
):

    (
        x1,
        y1,
        x2,
        y2,

    ) = box


    width = max(
        float(
            x2
            -
            x1
        ),
        1.0,
    )


    height = max(
        float(
            y2
            -
            y1
        ),
        1.0,
    )


    expanded_x1 = (
        x1
        -
        BOX_X_MARGIN_RATIO
        *
        width
    )


    expanded_x2 = (
        x2
        +
        BOX_X_MARGIN_RATIO
        *
        width
    )


    expanded_y1 = (
        y1
        -
        BOX_Y_MARGIN_RATIO
        *
        height
    )


    expanded_y2 = (
        y2
    )


    inside_expanded = (

        expanded_x1
        <=
        ball_x
        <=
        expanded_x2

        and

        expanded_y1
        <=
        ball_y
        <=
        expanded_y2
    )


    normalized_x = (

        (
            ball_x
            -
            x1
        )

        /

        width
    )


    normalized_y = (

        (
            ball_y
            -
            y1
        )

        /

        height
    )


    upper_body = (

        inside_expanded

        and

        normalized_y
        <=
        AIR_ZONE_Y_MAX
    )


    upper_center_x = (
        x1
        +
        0.5
        *
        width
    )


    upper_center_y = (
        y1
        +
        0.25
        *
        height
    )


    upper_distance = float(
        np.hypot(

            ball_x
            -
            upper_center_x,

            ball_y
            -
            upper_center_y,
        )
    )


    return {
        "inside_expanded":
            inside_expanded,

        "upper_body":
            upper_body,

        "normalized_x":
            float(
                normalized_x
            ),

        "normalized_y":
            float(
                normalized_y
            ),

        "upper_distance_px":
            upper_distance,
    }


# ============================================================
# AirTouch V3
# ============================================================

def run_v3(
    video_path,
    player_model_path,
    player_csv,
    possession_csv,
    device,
):

    (
        _,
        tracks_by_frame,
        _,
        possession_by_frame,

    ) = prepare_inputs(
        player_csv,
        possession_csv,
    )


    model = YOLO(
        str(
            player_model_path
        )
    )


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


    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )


    interaction_rows = []

    frame_idx = 0


    print("")
    print(
        "=" * 58
    )

    print(
        "AIR INTERACTION — PASS A / V3 EXTRACTION"
    )

    print(
        "=" * 58
    )


    while True:

        ret, frame = (
            cap.read()
        )


        if not ret:

            break


        possession_row = (
            get_possession_row(
                possession_by_frame,
                frame_idx,
            )
        )


        if (
            possession_row
            is None
        ):

            frame_idx += 1
            continue


        ball_state = str(
            possession_row.get(
                "ball_state_v4",
                "",
            )
        )


        if (
            ball_state
            not in
            ALLOWED_BALL_STATES
        ):

            frame_idx += 1
            continue


        ball_x = (
            possession_row.get(
                "ball_image_x",
                np.nan,
            )
        )


        ball_y = (
            possession_row.get(
                "ball_image_y",
                np.nan,
            )
        )


        if (
            pd.isna(
                ball_x
            )
            or
            pd.isna(
                ball_y
            )
        ):

            frame_idx += 1
            continue


        ball_x = float(
            ball_x
        )


        ball_y = float(
            ball_y
        )


        ball_speed = (
            possession_row.get(
                "ball_speed_mps",
                np.nan,
            )
        )


        detections = (
            detect_player_boxes(
                model,
                frame,
                device,
            )
        )


        frame_tracks = (
            tracks_by_frame.get(
                frame_idx,
                None,
            )
        )


        matched_players = (
            match_tracks_to_boxes(
                frame_tracks,
                detections,
            )
        )


        deflection = (
            calculate_deflection(
                possession_by_frame,
                frame_idx,
            )
        )


        strong_deflection = (
            is_strong_deflection(
                deflection
            )
        )


        very_strong = (
            is_very_strong_air_touch(
                ball_speed,
                deflection,
            )
        )


        for player in (
            matched_players
        ):

            zone = upper_body_metrics(

                ball_x,
                ball_y,

                player[
                    "box"
                ],
            )


            if not zone[
                "upper_body"
            ]:

                continue


            followed_by_control = (
                followed_by_same_player_control(

                    possession_by_frame,

                    frame_idx,

                    player[
                        "team"
                    ],

                    player[
                        "track_id"
                    ],
                )
            )


            (
                x1,
                y1,
                x2,
                y2,

            ) = player[
                "box"
            ]


            interaction_rows.append(
                {
                    "frame":
                        frame_idx,

                    "time_sec":
                        frame_idx
                        /
                        fps,

                    "team":
                        player[
                            "team"
                        ],

                    "track_id":
                        player[
                            "track_id"
                        ],

                    "ball_state_v4":
                        ball_state,

                    "possession_v4":
                        possession_row.get(
                            "possession_v4",
                            None,
                        ),

                    "ball_speed_mps":
                        ball_speed,

                    "ball_image_x":
                        ball_x,

                    "ball_image_y":
                        ball_y,

                    "bbox_x1":
                        float(
                            x1
                        ),

                    "bbox_y1":
                        float(
                            y1
                        ),

                    "bbox_x2":
                        float(
                            x2
                        ),

                    "bbox_y2":
                        float(
                            y2
                        ),

                    "normalized_ball_x":
                        zone[
                            "normalized_x"
                        ],

                    "normalized_ball_y":
                        zone[
                            "normalized_y"
                        ],

                    "upper_distance_px":
                        zone[
                            "upper_distance_px"
                        ],

                    "trajectory_valid":
                        deflection[
                            "valid"
                        ],

                    "direction_change_deg":
                        deflection[
                            "direction_change_deg"
                        ],

                    "speed_change_ratio":
                        deflection[
                            "speed_change_ratio"
                        ],

                    "strong_deflection":
                        strong_deflection,

                    "very_strong_air_touch":
                        very_strong,

                    "followed_by_same_player_control":
                        followed_by_control,

                    "box_match_distance_px":
                        player[
                            "match_distance_px"
                        ],

                    "box_confidence":
                        player[
                            "box_confidence"
                        ],
                }
            )


        if (
            frame_idx
            %
            100
            ==
            0
        ):

            print(
                f"frame={frame_idx:04d} | "
                f"interactions="
                f"{len(interaction_rows)}"
            )


        frame_idx += 1


    cap.release()


    interactions = pd.DataFrame(
        interaction_rows
    )


    interaction_columns = [
        "frame",
        "time_sec",

        "team",
        "track_id",

        "ball_state_v4",
        "possession_v4",

        "ball_speed_mps",

        "ball_image_x",
        "ball_image_y",

        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",

        "normalized_ball_x",
        "normalized_ball_y",

        "upper_distance_px",

        "trajectory_valid",

        "direction_change_deg",
        "speed_change_ratio",

        "strong_deflection",
        "very_strong_air_touch",

        "followed_by_same_player_control",

        "box_match_distance_px",
        "box_confidence",

        "event_id",
        "event_class",
        "event_confirmed_air_touch",
        "event_decision_reason",
    ]


    if len(
        interactions
    ) == 0:

        interactions = pd.DataFrame(
            columns=(
                interaction_columns
            )
        )


        empty_events = pd.DataFrame(
            columns=[
                "team",
                "track_id",

                "start_frame",
                "end_frame",

                "start_time_sec",
                "end_time_sec",

                "duration_frames",

                "interaction_frame_count",

                "max_direction_change_deg",
                "max_speed_change_ratio",
                "max_ball_speed_mps",

                "min_upper_distance_px",

                "has_strong_deflection",
                "has_very_strong_frame",
                "has_reception_evidence",

                "event_class",
                "confirmed_air_touch",
                "decision_reason",

                "event_id",
            ]
        )


        return (
            interactions,
            empty_events,
            fps,
        )


    interactions = (
        interactions
        .sort_values(
            [
                "team",
                "track_id",
                "frame",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # V3 event grouping
    # ========================================================

    temporary_events = []


    for (
        team,
        track_id,

    ), group in interactions.groupby(
        [
            "team",
            "track_id",
        ],
        sort=False,
    ):

        group = (
            group
            .sort_values(
                "frame"
            )
        )


        current_indices = []

        previous_frame = None


        for (
            row_index,
            row,

        ) in group.iterrows():

            frame = int(
                row[
                    "frame"
                ]
            )


            if (
                previous_frame
                is None
            ):

                current_indices = [
                    row_index
                ]


            elif (
                frame
                -
                previous_frame
                <=
                MAX_EVENT_FRAME_GAP
            ):

                current_indices.append(
                    row_index
                )


            else:

                temporary_events.append(
                    {
                        "team":
                            team,

                        "track_id":
                            float(
                                track_id
                            ),

                        "member_indices":
                            current_indices.copy(),
                    }
                )


                current_indices = [
                    row_index
                ]


            previous_frame = (
                frame
            )


        if len(
            current_indices
        ) > 0:

            temporary_events.append(
                {
                    "team":
                        team,

                    "track_id":
                        float(
                            track_id
                        ),

                    "member_indices":
                        current_indices.copy(),
                }
            )


    # ========================================================
    # V3 event features
    # ========================================================

    event_rows = []


    for event in (
        temporary_events
    ):

        member_df = (
            interactions.loc[
                event[
                    "member_indices"
                ]
            ]
            .copy()
        )


        start_frame = int(
            member_df[
                "frame"
            ]
            .min()
        )


        end_frame = int(
            member_df[
                "frame"
            ]
            .max()
        )


        has_strong = bool(
            member_df[
                "strong_deflection"
            ]
            .any()
        )


        has_very_strong = bool(
            member_df[
                "very_strong_air_touch"
            ]
            .any()
        )


        has_reception_evidence = bool(
            member_df[
                "followed_by_same_player_control"
            ]
            .any()
        )


        valid_direction = pd.to_numeric(
            member_df[
                "direction_change_deg"
            ],
            errors="coerce",
        )


        valid_speed_change = pd.to_numeric(
            member_df[
                "speed_change_ratio"
            ],
            errors="coerce",
        )


        valid_ball_speed = pd.to_numeric(
            member_df[
                "ball_speed_mps"
            ],
            errors="coerce",
        )


        max_direction = (

            valid_direction.max()

            if
            valid_direction.notna().any()

            else
            np.nan
        )


        max_speed_change = (

            valid_speed_change.max()

            if
            valid_speed_change.notna().any()

            else
            np.nan
        )


        max_ball_speed = (

            valid_ball_speed.max()

            if
            valid_ball_speed.notna().any()

            else
            np.nan
        )


        min_upper_distance = (
            member_df[
                "upper_distance_px"
            ]
            .min()
        )


        # ====================================================
        # Exact V3 event classification priority
        # ====================================================

        if has_very_strong:

            event_class = (
                "AirTouch"
            )

            confirmed_air_touch = True

            decision_reason = (
                "very_strong_air_touch"
            )


        elif (
            has_reception_evidence
        ):

            event_class = (
                "ReceptionOrBounce"
            )

            confirmed_air_touch = False

            decision_reason = (
                "event_contains_same_player_reception"
            )


        elif has_strong:

            event_class = (
                "AirTouch"
            )

            confirmed_air_touch = True

            decision_reason = (
                "strong_deflection_without_reception"
            )


        else:

            event_class = (
                "WeakInteraction"
            )

            confirmed_air_touch = False

            decision_reason = (
                "weak_deflection"
            )


        event_rows.append(
            {
                "team":
                    event[
                        "team"
                    ],

                "track_id":
                    event[
                        "track_id"
                    ],

                "start_frame":
                    start_frame,

                "end_frame":
                    end_frame,

                "start_time_sec":
                    start_frame
                    /
                    fps,

                "end_time_sec":
                    end_frame
                    /
                    fps,

                "duration_frames":
                    end_frame
                    -
                    start_frame
                    +
                    1,

                "interaction_frame_count":
                    len(
                        member_df
                    ),

                "max_direction_change_deg":
                    max_direction,

                "max_speed_change_ratio":
                    max_speed_change,

                "max_ball_speed_mps":
                    max_ball_speed,

                "min_upper_distance_px":
                    min_upper_distance,

                "has_strong_deflection":
                    has_strong,

                "has_very_strong_frame":
                    has_very_strong,

                "has_reception_evidence":
                    has_reception_evidence,

                "event_class":
                    event_class,

                "confirmed_air_touch":
                    confirmed_air_touch,

                "decision_reason":
                    decision_reason,

                "_member_indices":
                    event[
                        "member_indices"
                    ],
            }
        )


    events = pd.DataFrame(
        event_rows
    )


    events = (
        events
        .sort_values(
            [
                "start_frame",
                "team",
                "track_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    events[
        "event_id"
    ] = np.arange(
        1,
        len(
            events
        )
        +
        1,
    )


    # ========================================================
    # Map event decision back to interaction rows
    # ========================================================

    interactions[
        "event_id"
    ] = np.nan


    interactions[
        "event_class"
    ] = None


    interactions[
        "event_confirmed_air_touch"
    ] = False


    interactions[
        "event_decision_reason"
    ] = None


    for (
        _,
        event,

    ) in events.iterrows():

        member_indices = (
            event[
                "_member_indices"
            ]
        )


        interactions.loc[
            member_indices,
            "event_id",
        ] = int(
            event[
                "event_id"
            ]
        )


        interactions.loc[
            member_indices,
            "event_class",
        ] = (
            event[
                "event_class"
            ]
        )


        interactions.loc[
            member_indices,
            "event_confirmed_air_touch",
        ] = bool(
            event[
                "confirmed_air_touch"
            ]
        )


        interactions.loc[
            member_indices,
            "event_decision_reason",
        ] = (
            event[
                "decision_reason"
            ]
        )


    events_output = (
        events
        .drop(
            columns=[
                "_member_indices"
            ]
        )
        .copy()
    )


    return (
        interactions,
        events_output,
        fps,
    )


# ============================================================
# AirTouch V4 cross-player arbitration
# ============================================================

def run_v4(
    interactions,
    events,
):

    interactions = (
        interactions.copy()
    )


    events = (
        events.copy()
    )


    interaction_numeric = [
        "frame",
        "event_id",
        "track_id",

        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",

        "upper_distance_px",

        "ball_speed_mps",
        "direction_change_deg",
    ]


    for column in (
        interaction_numeric
    ):

        if (
            column
            in
            interactions.columns
        ):

            interactions[
                column
            ] = pd.to_numeric(
                interactions[
                    column
                ],
                errors="coerce",
            )


    event_numeric = [
        "event_id",
        "track_id",

        "start_frame",
        "end_frame",

        "start_time_sec",
        "end_time_sec",

        "max_direction_change_deg",
        "max_speed_change_ratio",
        "max_ball_speed_mps",

        "min_upper_distance_px",
    ]


    for column in (
        event_numeric
    ):

        if (
            column
            in
            events.columns
        ):

            events[
                column
            ] = pd.to_numeric(
                events[
                    column
                ],
                errors="coerce",
            )


    for column in [
        "confirmed_air_touch",
        "has_very_strong_frame",
        "has_reception_evidence",
    ]:

        if (
            column
            in
            events.columns
        ):

            events[
                column
            ] = (
                events[
                    column
                ]
                .apply(
                    as_bool
                )
            )


    # ========================================================
    # Normalized upper-body distance
    # ========================================================

    interactions[
        "bbox_height"
    ] = (

        interactions[
            "bbox_y2"
        ]

        -

        interactions[
            "bbox_y1"
        ]
    )


    interactions[
        "upper_distance_norm"
    ] = (

        interactions[
            "upper_distance_px"
        ]

        /

        interactions[
            "bbox_height"
        ]
        .clip(
            lower=1.0
        )
    )


    event_distance_norm = (

        interactions
        .groupby(
            "event_id"
        )[
            "upper_distance_norm"
        ]
        .min()
    )


    events[
        "min_upper_distance_norm"
    ] = (

        events[
            "event_id"
        ]
        .map(
            event_distance_norm
        )
    )


    # ========================================================
    # Split V3 events
    # ========================================================

    air_candidates = (
        events[
            events[
                "confirmed_air_touch"
            ]
            ==
            True
        ]
        .copy()
    )


    non_air_events = (
        events[
            events[
                "confirmed_air_touch"
            ]
            ==
            False
        ]
        .copy()
    )


    air_candidates = (
        air_candidates
        .sort_values(
            [
                "start_frame",
                "end_frame",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Group overlapping AirTouch candidates
    # ========================================================

    candidate_groups = []

    current_group = []
    current_end = None


    for (
        _,
        row,

    ) in air_candidates.iterrows():

        start_frame = int(
            row[
                "start_frame"
            ]
        )


        end_frame = int(
            row[
                "end_frame"
            ]
        )


        if not current_group:

            current_group = [
                row
            ]

            current_end = (
                end_frame
            )

            continue


        if (
            start_frame
            <=
            current_end
            +
            ARBITRATION_FRAME_GAP
        ):

            current_group.append(
                row
            )


            current_end = max(
                current_end,
                end_frame,
            )


        else:

            candidate_groups.append(
                current_group
            )


            current_group = [
                row
            ]


            current_end = (
                end_frame
            )


    if current_group:

        candidate_groups.append(
            current_group
        )


    physical_rows = []


    # ========================================================
    # Preserve V3 non-AirTouch events
    # ========================================================

    for (
        _,
        row,

    ) in non_air_events.iterrows():

        physical_rows.append(
            {
                "start_frame":
                    int(
                        row[
                            "start_frame"
                        ]
                    ),

                "end_frame":
                    int(
                        row[
                            "end_frame"
                        ]
                    ),

                "start_time_sec":
                    float(
                        row[
                            "start_time_sec"
                        ]
                    ),

                "end_time_sec":
                    float(
                        row[
                            "end_time_sec"
                        ]
                    ),

                "final_class":
                    str(
                        row[
                            "event_class"
                        ]
                    ),

                "last_touch_team":
                    None,

                "last_touch_track_id":
                    np.nan,

                "participants":
                    participant_label(
                        row[
                            "team"
                        ],
                        row[
                            "track_id"
                        ],
                    ),

                "source_v3_event_ids":
                    str(
                        int(
                            row[
                                "event_id"
                            ]
                        )
                    ),

                "candidate_count":
                    1,

                "max_ball_speed_mps":
                    row[
                        "max_ball_speed_mps"
                    ],

                "max_direction_change_deg":
                    row[
                        "max_direction_change_deg"
                    ],

                "min_upper_distance_norm":
                    row[
                        "min_upper_distance_norm"
                    ],

                "decision_reason":
                    str(
                        row[
                            "decision_reason"
                        ]
                    ),
            }
        )


    # ========================================================
    # Cross-player arbitration
    # ========================================================

    for group in (
        candidate_groups
    ):

        group_df = pd.DataFrame(
            group
        ).copy()


        group_df = (
            group_df
            .sort_values(
                [
                    "min_upper_distance_norm",
                    "event_id",
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )


        start_frame = int(
            group_df[
                "start_frame"
            ]
            .min()
        )


        end_frame = int(
            group_df[
                "end_frame"
            ]
            .max()
        )


        start_time = float(
            group_df[
                "start_time_sec"
            ]
            .min()
        )


        end_time = float(
            group_df[
                "end_time_sec"
            ]
            .max()
        )


        participants = "|".join(
            [
                participant_label(
                    row[
                        "team"
                    ],
                    row[
                        "track_id"
                    ],
                )

                for (
                    _,
                    row,
                )
                in group_df.iterrows()
            ]
        )


        source_event_ids = ";".join(
            [
                str(
                    int(
                        event_id
                    )
                )

                for event_id
                in group_df[
                    "event_id"
                ]
                .tolist()
            ]
        )


        teams = set(
            group_df[
                "team"
            ]
            .astype(str)
        )


        max_ball_speed = (
            group_df[
                "max_ball_speed_mps"
            ]
            .max()
        )


        max_direction = (
            group_df[
                "max_direction_change_deg"
            ]
            .max()
        )


        candidate_count = (
            len(
                group_df
            )
        )


        # ====================================================
        # Case 1: one candidate
        # ====================================================

        if (
            candidate_count
            ==
            1
        ):

            winner = (
                group_df.iloc[
                    0
                ]
            )


            final_class = (
                "AirTouch"
            )


            last_touch_team = (
                winner[
                    "team"
                ]
            )


            last_touch_track_id = (
                winner[
                    "track_id"
                ]
            )


            decision_reason = (
                "single_air_touch_candidate"
            )


            min_distance_norm = (
                winner[
                    "min_upper_distance_norm"
                ]
            )


        # ====================================================
        # Case 2: multiple candidates, same team
        # ====================================================

        elif (
            len(
                teams
            )
            ==
            1
        ):

            winner = (
                group_df.iloc[
                    0
                ]
            )


            final_class = (
                "AirTouch"
            )


            last_touch_team = (
                winner[
                    "team"
                ]
            )


            last_touch_track_id = (
                winner[
                    "track_id"
                ]
            )


            decision_reason = (
                "same_team_overlap_choose_nearest"
            )


            min_distance_norm = (
                winner[
                    "min_upper_distance_norm"
                ]
            )


        # ====================================================
        # Case 3: opposite-team candidates
        # ====================================================

        else:

            group_df[
                "_evidence_rank"
            ] = (
                group_df[
                    "has_very_strong_frame"
                ]
                .apply(
                    lambda value:
                        2
                        if as_bool(
                            value
                        )
                        else 1
                )
            )


            group_df = (
                group_df
                .sort_values(
                    [
                        "_evidence_rank",
                        "min_upper_distance_norm",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                    na_position="last",
                )
                .reset_index(
                    drop=True
                )
            )


            best = (
                group_df.iloc[
                    0
                ]
            )


            second = (
                group_df.iloc[
                    1
                ]
            )


            best_rank = int(
                best[
                    "_evidence_rank"
                ]
            )


            second_rank = int(
                second[
                    "_evidence_rank"
                ]
            )


            best_distance = (
                best[
                    "min_upper_distance_norm"
                ]
            )


            second_distance = (
                second[
                    "min_upper_distance_norm"
                ]
            )


            # ------------------------------------------------
            # Stronger evidence class wins
            # ------------------------------------------------

            if (
                best_rank
                >
                second_rank
            ):

                final_class = (
                    "AirTouch"
                )


                last_touch_team = (
                    best[
                        "team"
                    ]
                )


                last_touch_track_id = (
                    best[
                        "track_id"
                    ]
                )


                decision_reason = (
                    "unique_stronger_cross_team_candidate"
                )


                min_distance_norm = (
                    best_distance
                )


            # ------------------------------------------------
            # Same evidence rank:
            # require very clear spatial winner
            # ------------------------------------------------

            else:

                clear_spatial_winner = (
                    False
                )


                if (
                    pd.notna(
                        best_distance
                    )

                    and

                    pd.notna(
                        second_distance
                    )

                    and

                    second_distance
                    >
                    0
                ):

                    ratio = (
                        best_distance
                        /
                        second_distance
                    )


                    margin = (
                        second_distance
                        -
                        best_distance
                    )


                    clear_spatial_winner = (

                        ratio
                        <=
                        CLEAR_WINNER_DISTANCE_RATIO

                        and

                        margin
                        >=
                        CLEAR_WINNER_DISTANCE_MARGIN_NORM
                    )


                if clear_spatial_winner:

                    final_class = (
                        "AirTouch"
                    )


                    last_touch_team = (
                        best[
                            "team"
                        ]
                    )


                    last_touch_track_id = (
                        best[
                            "track_id"
                        ]
                    )


                    decision_reason = (
                        "clear_spatial_winner_cross_team"
                    )


                    min_distance_norm = (
                        best_distance
                    )


                else:

                    final_class = (
                        "AerialContest"
                    )


                    last_touch_team = (
                        None
                    )


                    last_touch_track_id = (
                        np.nan
                    )


                    decision_reason = (
                        "cross_team_air_touch_ambiguity"
                    )


                    min_distance_norm = (
                        group_df[
                            "min_upper_distance_norm"
                        ]
                        .min()
                    )


        physical_rows.append(
            {
                "start_frame":
                    start_frame,

                "end_frame":
                    end_frame,

                "start_time_sec":
                    start_time,

                "end_time_sec":
                    end_time,

                "final_class":
                    final_class,

                "last_touch_team":
                    last_touch_team,

                "last_touch_track_id":
                    last_touch_track_id,

                "participants":
                    participants,

                "source_v3_event_ids":
                    source_event_ids,

                "candidate_count":
                    candidate_count,

                "max_ball_speed_mps":
                    max_ball_speed,

                "max_direction_change_deg":
                    max_direction,

                "min_upper_distance_norm":
                    min_distance_norm,

                "decision_reason":
                    decision_reason,
            }
        )


    column_order = [
        "physical_event_id",

        "start_frame",
        "end_frame",

        "start_time_sec",
        "end_time_sec",

        "final_class",

        "last_touch_team",
        "last_touch_track_id",

        "participants",

        "candidate_count",

        "max_ball_speed_mps",
        "max_direction_change_deg",

        "min_upper_distance_norm",

        "decision_reason",

        "source_v3_event_ids",
    ]


    if len(
        physical_rows
    ) == 0:

        return pd.DataFrame(
            columns=(
                column_order
            )
        )


    final_events = pd.DataFrame(
        physical_rows
    )


    final_events = (
        final_events
        .sort_values(
            [
                "start_frame",
                "end_frame",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    final_events[
        "physical_event_id"
    ] = np.arange(
        1,
        len(
            final_events
        )
        +
        1,
    )


    return (
        final_events[
            column_order
        ]
    )


# ============================================================
# Regression helpers
# ============================================================

def find_v3_event(
    events,
    frame,
    team,
    track_id,
):

    matches = events[

        (
            events[
                "team"
            ]
            .astype(str)
            ==
            str(
                team
            )
        )

        &

        (
            np.abs(
                events[
                    "track_id"
                ]
                .astype(float)

                -

                float(
                    track_id
                )
            )
            <
            0.1
        )

        &

        (
            events[
                "start_frame"
            ]
            <=
            frame
        )

        &

        (
            events[
                "end_frame"
            ]
            >=
            frame
        )
    ]


    if len(
        matches
    ) == 0:

        return None


    return (
        matches.iloc[
            0
        ]
    )


def final_events_at_frame(
    final_events,
    frame,
):

    return final_events[

        (
            final_events[
                "start_frame"
            ]
            <=
            frame
        )

        &

        (
            final_events[
                "end_frame"
            ]
            >=
            frame
        )
    ]


def run_regression_checks(
    v3_events,
    final_events,
):

    negative = find_v3_event(
        v3_events,
        NEGATIVE_FRAME,
        NEGATIVE_TEAM,
        NEGATIVE_TRACK_ID,
    )


    positive = find_v3_event(
        v3_events,
        POSITIVE_FRAME,
        POSITIVE_TEAM,
        POSITIVE_TRACK_ID,
    )


    v3_negative_pass = (

        negative
        is not None

        and

        not as_bool(
            negative[
                "confirmed_air_touch"
            ]
        )
    )


    v3_positive_pass = (

        positive
        is not None

        and

        as_bool(
            positive[
                "confirmed_air_touch"
            ]
        )
    )


    contest_pass = False


    contest_rows = (
        final_events_at_frame(
            final_events,
            CONTEST_FRAME,
        )
    )


    for (
        _,
        row,

    ) in contest_rows.iterrows():

        if (
            row[
                "final_class"
            ]
            ==
            "AerialContest"
        ):

            participants = str(
                row[
                    "participants"
                ]
            )


            if (
                "A#1"
                in
                participants

                and

                "B#14"
                in
                participants
            ):

                contest_pass = True
                break


    reception_pass = False


    reception_rows = (
        final_events_at_frame(
            final_events,
            RECEPTION_FRAME,
        )
    )


    for (
        _,
        row,

    ) in reception_rows.iterrows():

        if (

            row[
                "final_class"
            ]
            ==
            "ReceptionOrBounce"

            and

            participant_label(
                RECEPTION_TEAM,
                RECEPTION_TRACK_ID,
            )
            in
            str(
                row[
                    "participants"
                ]
            )
        ):

            reception_pass = True
            break


    header_pass = False


    header_rows = (
        final_events_at_frame(
            final_events,
            HEADER_FRAME,
        )
    )


    for (
        _,
        row,

    ) in header_rows.iterrows():

        if (

            row[
                "final_class"
            ]
            ==
            "AirTouch"

            and

            str(
                row[
                    "last_touch_team"
                ]
            )
            ==
            HEADER_TEAM

            and

            same_player_id(
                row[
                    "last_touch_track_id"
                ],
                HEADER_TRACK_ID,
            )
        ):

            header_pass = True
            break


    return {
        "v3_negative_203":
            v3_negative_pass,

        "v3_positive_281":
            v3_positive_pass,

        "v4_contest_172":
            contest_pass,

        "v4_reception_203":
            reception_pass,

        "v4_header_281":
            header_pass,
    }


# ============================================================
# Final QA video
# ============================================================

def render_qa_video(
    video_path,
    interactions,
    final_events,
    output_video,
):

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video for QA: "
            f"{video_path}"
        )


    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
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


    writer = cv2.VideoWriter(

        str(
            output_video
        ),

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

        cap.release()

        raise RuntimeError(
            f"Cannot create QA video: "
            f"{output_video}"
        )


    interaction_lookup = {}


    if len(
        interactions
    ) > 0:

        for (
            frame,
            event_id,

        ), group in interactions.groupby(
            [
                "frame",
                "event_id",
            ]
        ):

            if pd.isna(
                event_id
            ):

                continue


            interaction_lookup[
                (
                    int(
                        frame
                    ),

                    int(
                        event_id
                    ),
                )
            ] = (
                group.copy()
            )


    final_events_by_frame = {}


    for (
        _,
        event,

    ) in final_events.iterrows():

        start = int(
            event[
                "start_frame"
            ]
        )


        end = int(
            event[
                "end_frame"
            ]
        )


        for frame in range(
            start,
            end + 1,
        ):

            final_events_by_frame.setdefault(
                frame,
                [],
            ).append(
                event
            )


    frame_idx = 0


    while True:

        ret, frame = (
            cap.read()
        )


        if not ret:

            break


        annotated = (
            frame.copy()
        )


        active_events = (
            final_events_by_frame.get(
                frame_idx,
                [],
            )
        )


        y_overlay = 35


        for event in (
            active_events
        ):

            final_class = str(
                event[
                    "final_class"
                ]
            )


            if (
                final_class
                ==
                "AirTouch"
            ):

                color = (
                    0,
                    255,
                    0,
                )


            elif (
                final_class
                ==
                "AerialContest"
            ):

                color = (
                    255,
                    0,
                    255,
                )


            elif (
                final_class
                ==
                "ReceptionOrBounce"
            ):

                color = (
                    0,
                    165,
                    255,
                )


            else:

                color = (
                    180,
                    180,
                    180,
                )


            source_ids = (
                parse_event_ids(
                    event[
                        "source_v3_event_ids"
                    ]
                )
            )


            for source_event_id in (
                source_ids
            ):

                key = (
                    frame_idx,
                    source_event_id,
                )


                if (
                    key
                    not in
                    interaction_lookup
                ):

                    continue


                source_rows = (
                    interaction_lookup[
                        key
                    ]
                )


                for (
                    _,
                    source_row,

                ) in source_rows.iterrows():

                    x1 = int(
                        source_row[
                            "bbox_x1"
                        ]
                    )


                    y1 = int(
                        source_row[
                            "bbox_y1"
                        ]
                    )


                    x2 = int(
                        source_row[
                            "bbox_x2"
                        ]
                    )


                    y2 = int(
                        source_row[
                            "bbox_y2"
                        ]
                    )


                    cv2.rectangle(
                        annotated,

                        (
                            x1,
                            y1,
                        ),

                        (
                            x2,
                            y2,
                        ),

                        color,

                        3,
                    )


                    cv2.putText(
                        annotated,

                        participant_label(
                            source_row[
                                "team"
                            ],
                            source_row[
                                "track_id"
                            ],
                        ),

                        (
                            x1,

                            max(
                                22,
                                y1 - 8,
                            ),
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.48,

                        color,

                        2,

                        cv2.LINE_AA,
                    )


            if (
                final_class
                ==
                "AirTouch"
            ):

                event_text = (

                    f"AIR TOUCH | "
                    f"LAST TOUCH "
                    f"{event['last_touch_team']} "
                    f"#{int(event['last_touch_track_id'])}"
                )


            elif (
                final_class
                ==
                "AerialContest"
            ):

                event_text = (

                    f"AERIAL CONTEST | "
                    f"{event['participants']} | "
                    f"LAST TOUCH UNKNOWN"
                )


            else:

                event_text = (

                    f"{final_class} | "
                    f"{event['participants']}"
                )


            cv2.putText(
                annotated,

                event_text,

                (
                    20,
                    y_overlay,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.62,

                color,

                2,

                cv2.LINE_AA,
            )


            y_overlay += 30


        cv2.putText(
            annotated,

            (
                f"AIR INTERACTION | "
                f"frame={frame_idx} | "
                f"time="
                f"{frame_idx / fps:.2f}s"
            ),

            (
                20,
                height - 25,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (
                255,
                255,
                255,
            ),

            2,

            cv2.LINE_AA,
        )


        writer.write(
            annotated
        )


        frame_idx += 1


    cap.release()

    writer.release()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Canonical air interaction stage: "
            "AirTouch V3 extraction + "
            "AirTouch V4 arbitration."
        )
    )


    parser.add_argument(
        "--video",
        required=True,
    )


    parser.add_argument(
        "--player-model",
        required=True,
    )


    parser.add_argument(
        "--players",
        required=True,
    )


    parser.add_argument(
        "--possession",
        required=True,
    )


    parser.add_argument(
        "--output-interactions",
        required=True,
    )


    parser.add_argument(
        "--output-v3-events",
        required=True,
    )


    parser.add_argument(
        "--output",
        required=True,
    )


    parser.add_argument(
        "--output-v3-confirmed",
        default=None,
    )


    parser.add_argument(
        "--output-confirmed",
        default=None,
    )


    parser.add_argument(
        "--output-contests",
        default=None,
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
        "--skip-qa",
        action="store_true",
    )


    args = (
        parser.parse_args()
    )


    video_path = Path(
        args.video
    )


    player_model_path = Path(
        args.player_model
    )


    player_csv = Path(
        args.players
    )


    possession_csv = Path(
        args.possession
    )


    output_interactions = Path(
        args.output_interactions
    )


    output_v3_events = Path(
        args.output_v3_events
    )


    output_final = Path(
        args.output
    )


    require_file(
        video_path,
        "Input video",
    )


    require_file(
        player_model_path,
        "Player model",
    )


    require_file(
        player_csv,
        "Clean player tracking CSV",
    )


    require_file(
        possession_csv,
        "Possession seed CSV",
    )


    for path in [
        output_interactions,
        output_v3_events,
        output_final,
    ]:

        ensure_parent(
            path
        )


    optional_paths = [
        args.output_v3_confirmed,
        args.output_confirmed,
        args.output_contests,
        args.output_video,
    ]


    for value in (
        optional_paths
    ):

        if value:

            ensure_parent(
                Path(
                    value
                )
            )


    device = resolve_device(
        args.device
    )


    print("")
    print(
        "=" * 62
    )

    print(
        "AIR INTERACTION CLI"
    )

    print(
        "=" * 62
    )


    print(
        "Video:",
        video_path,
    )


    print(
        "Player tracks:",
        player_csv,
    )


    print(
        "Possession seed:",
        possession_csv,
    )


    print(
        "Player model:",
        player_model_path,
    )


    print(
        "Device:",
        device,
    )


    # ========================================================
    # PASS A — AirTouch V3
    # ========================================================

    (
        interactions,
        v3_events,
        fps,

    ) = run_v3(

        video_path=(
            video_path
        ),

        player_model_path=(
            player_model_path
        ),

        player_csv=(
            player_csv
        ),

        possession_csv=(
            possession_csv
        ),

        device=(
            device
        ),
    )


    interactions.to_csv(
        output_interactions,
        index=False,
    )


    v3_events.to_csv(
        output_v3_events,
        index=False,
    )


    if (
        args.output_v3_confirmed
    ):

        v3_confirmed = (
            v3_events[
                v3_events[
                    "confirmed_air_touch"
                ]
                .apply(
                    as_bool
                )
            ]
            .copy()
        )


        v3_confirmed.to_csv(
            args.output_v3_confirmed,
            index=False,
        )


    print("")
    print(
        "V3 interaction rows:",
        len(
            interactions
        ),
    )


    print(
        "V3 events:",
        len(
            v3_events
        ),
    )


    if len(
        v3_events
    ):

        print("")
        print(
            "V3 CLASSES"
        )


        print(
            v3_events[
                "event_class"
            ]
            .value_counts()
            .to_string()
        )


    # ========================================================
    # PASS B — AirTouch V4
    # ========================================================

    print("")
    print(
        "=" * 58
    )

    print(
        "AIR INTERACTION — PASS B / V4 ARBITRATION"
    )

    print(
        "=" * 58
    )


    final_events = run_v4(
        interactions,
        v3_events,
    )


    final_events.to_csv(
        output_final,
        index=False,
    )


    if (
        args.output_confirmed
    ):

        (
            final_events[
                final_events[
                    "final_class"
                ]
                ==
                "AirTouch"
            ]
            .copy()
            .to_csv(
                args.output_confirmed,
                index=False,
            )
        )


    if (
        args.output_contests
    ):

        (
            final_events[
                final_events[
                    "final_class"
                ]
                ==
                "AerialContest"
            ]
            .copy()
            .to_csv(
                args.output_contests,
                index=False,
            )
        )


    checks = (
        run_regression_checks(
            v3_events,
            final_events,
        )
    )


    print(
        "V4 physical events:",
        len(
            final_events
        ),
    )


    if len(
        final_events
    ):

        print("")
        print(
            "FINAL CLASSES"
        )


        print(
            final_events[
                "final_class"
            ]
            .value_counts()
            .to_string()
        )


    print("")
    print(
        "REGRESSION CHECKS"
    )


    for (
        name,
        passed,

    ) in checks.items():

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"{name}"
        )


    # ========================================================
    # QA
    # ========================================================

    if (
        args.output_video
        and
        not args.skip_qa
    ):

        print("")
        print(
            "Rendering QA video..."
        )


        render_qa_video(

            video_path=(
                video_path
            ),

            interactions=(
                interactions
            ),

            final_events=(
                final_events
            ),

            output_video=(
                Path(
                    args.output_video
                )
            ),
        )


        print(
            "QA video:",
            args.output_video,
        )


    print("")
    print(
        "=" * 62
    )

    print(
        "DONE"
    )

    print(
        "=" * 62
    )


    print(
        "FPS:",
        f"{fps:.3f}",
    )


    print(
        "Interactions:",
        output_interactions,
    )


    print(
        "V3 events:",
        output_v3_events,
    )


    print(
        "Final events:",
        output_final,
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
