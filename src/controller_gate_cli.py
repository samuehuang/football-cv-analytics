#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO


# ============================================================
# Validated Controller Gate V2 parameters
# ============================================================

GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2

PLAYER_CONF = 0.25
PLAYER_IMGSZ = 1280

MIN_DOMINANT_TEAM_SHARE = 0.70

MAX_TRACK_BOX_MATCH_PX = 55.0

CONTEST_POST_LOCK_FRAMES = 4

HARD_MAX_FOOT_DISTANCE_NORM = 1.35

HARD_MIN_BALL_REL_Y = 0.30
HARD_MAX_BALL_REL_Y = 1.45

CLOSE_CONTROL_MAX_FOOT_DISTANCE_NORM = 1.00

CLOSE_CONTROL_MIN_REL_Y = 0.50
CLOSE_CONTROL_MAX_REL_Y = 1.35

DRIBBLE_MAX_FOOT_DISTANCE_NORM = 1.20

DRIBBLE_MIN_REL_Y = 0.70
DRIBBLE_MAX_REL_Y = 1.20

MIN_DRIBBLE_CONTROLLER_RUN = 3

SMALL_BBOX_HEIGHT_PX = 35.0
SMALL_BBOX_MAX_FOOT_DISTANCE_PX = 38.0

SMALL_BBOX_MIN_REL_Y = 0.70
SMALL_BBOX_MAX_REL_Y = 1.20


# ============================================================
# Regression frames
# ============================================================

REGRESSION_CONTEST_FRAME = 175
REGRESSION_FALSE_CONTROL_FRAME = 226
REGRESSION_DRIBBLE_FRAME = 650


# ============================================================
# Helpers
# ============================================================

def require_file(path, label):

    path = Path(path)

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if path.stat().st_size == 0:

        raise RuntimeError(
            f"{label} is empty: {path}"
        )


def ensure_parent(path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def resolve_device(value):

    if value != "auto":
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


def clean_team(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {
        "A",
        "B",
    }:
        return value

    return None


def clean_id(value):

    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def same_player_id(a, b):

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


# ============================================================
# Normalize player tracks
# ============================================================

def prepare_tracks(
    tracks,
):

    required = [
        "frame",
        "track_id",
        "image_x",
        "image_y",
        "team",
    ]

    missing = [
        column

        for column
        in required

        if column
        not in tracks.columns
    ]

    if missing:

        raise ValueError(
            "Player CSV missing columns: "
            +
            ", ".join(missing)
        )


    for column in [
        "frame",
        "track_id",
        "image_x",
        "image_y",
    ]:

        tracks[column] = pd.to_numeric(
            tracks[column],
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


    tracks["frame"] = (
        tracks["frame"]
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


    stable_team_map = {}


    for (
        track_id,
        group,

    ) in tracks.groupby(
        "track_id"
    ):

        counts = (
            group["team"]
            .value_counts()
        )


        if len(counts) == 0:
            continue


        dominant_team = (
            counts.index[0]
        )


        dominant_share = (
            counts.iloc[0]
            /
            len(group)
        )


        if (
            dominant_share
            >=
            MIN_DOMINANT_TEAM_SHARE
        ):

            stable_team_map[
                track_id
            ] = dominant_team


    tracks["stable_team"] = (
        tracks["track_id"]
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


    # Preserve V2 behavior:
    # do not relabel contaminated rows.
    tracks = (
        tracks[
            tracks["team"]
            ==
            tracks["stable_team"]
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


    return (
        tracks,
        tracks_by_frame,
    )


# ============================================================
# Normalize possession + controller runs
# ============================================================

def prepare_possession(
    possession,
):

    required = [
        "frame",
        "ball_state_v5",
        "controller_team_v5",
        "controller_id_v5",
        "ball_image_x",
        "ball_image_y",
    ]


    missing = [
        column

        for column
        in required

        if column
        not in possession.columns
    ]


    if missing:

        raise ValueError(
            "Possession V5.1 CSV missing columns: "
            +
            ", ".join(missing)
        )


    for column in [
        "frame",
        "time_sec",
        "ball_image_x",
        "ball_image_y",
        "ball_speed_mps",
        "controller_id_v5",
    ]:

        if column in possession.columns:

            possession[column] = pd.to_numeric(
                possession[column],
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


    possession["frame"] = (
        possession["frame"]
        .astype(int)
    )


    possession = (
        possession
        .sort_values("frame")
        .reset_index(drop=True)
    )


    possession[
        "controller_run_id"
    ] = np.nan


    possession[
        "controller_run_length"
    ] = 0


    possession[
        "controller_run_position"
    ] = 0


    run_groups = []

    current_indices = []
    current_team = None
    current_id = np.nan
    previous_frame = None


    def flush_run():

        nonlocal current_indices
        nonlocal current_team
        nonlocal current_id
        nonlocal previous_frame

        if len(current_indices) > 0:

            run_groups.append(
                {
                    "indices":
                        current_indices.copy(),

                    "team":
                        current_team,

                    "track_id":
                        current_id,
                }
            )

        current_indices = []
        current_team = None
        current_id = np.nan
        previous_frame = None


    for (
        index,
        row,

    ) in possession.iterrows():

        frame = int(
            row["frame"]
        )


        state = str(
            row.get(
                "ball_state_v5",
                "",
            )
        )


        team = clean_team(
            row.get(
                "controller_team_v5",
                None,
            )
        )


        track_id = clean_id(
            row.get(
                "controller_id_v5",
                np.nan,
            )
        )


        valid_control = (

            state
            ==
            "Controlled"

            and

            team
            in {
                "A",
                "B",
            }

            and

            pd.notna(
                track_id
            )
        )


        if not valid_control:

            flush_run()
            continue


        if len(
            current_indices
        ) == 0:

            current_indices = [
                index
            ]

            current_team = team
            current_id = track_id
            previous_frame = frame

            continue


        same_controller = (

            team
            ==
            current_team

            and

            same_player_id(
                track_id,
                current_id,
            )
        )


        consecutive = (

            previous_frame
            is not None

            and

            frame
            ==
            previous_frame + 1
        )


        if (
            same_controller
            and
            consecutive
        ):

            current_indices.append(
                index
            )

            previous_frame = frame

        else:

            flush_run()

            current_indices = [
                index
            ]

            current_team = team
            current_id = track_id
            previous_frame = frame


    flush_run()


    for (
        run_id,
        run,

    ) in enumerate(
        run_groups,
        start=1,
    ):

        indices = run[
            "indices"
        ]

        run_length = len(
            indices
        )


        for (
            position,
            index,

        ) in enumerate(
            indices,
            start=1,
        ):

            possession.loc[
                index,
                "controller_run_id",
            ] = run_id


            possession.loc[
                index,
                "controller_run_length",
            ] = run_length


            possession.loc[
                index,
                "controller_run_position",
            ] = position


    possession_by_frame = (
        possession
        .set_index(
            "frame"
        )
    )


    controlled = (
        possession[
            possession[
                "ball_state_v5"
            ]
            ==
            "Controlled"
        ]
        .copy()
    )


    controlled_frames = set(
        controlled["frame"]
        .astype(int)
        .tolist()
    )


    return (
        possession,
        possession_by_frame,
        controlled_frames,
        run_groups,
    )


# ============================================================
# Contest lock
# ============================================================

def prepare_contest_lock(
    events,
):

    for column in [
        "physical_event_id",
        "start_frame",
        "end_frame",
    ]:

        if column in events.columns:

            events[column] = pd.to_numeric(
                events[column],
                errors="coerce",
            )


    events = (
        events
        .dropna(
            subset=[
                "start_frame",
                "end_frame",
                "final_class",
            ]
        )
        .copy()
    )


    events["start_frame"] = (
        events["start_frame"]
        .astype(int)
    )


    events["end_frame"] = (
        events["end_frame"]
        .astype(int)
    )


    contest_lock_frames = set()

    contest_source_map = {}


    physical_contests = (
        events[
            events[
                "final_class"
            ]
            ==
            "AerialContest"
        ]
        .copy()
    )


    for _, event in (
        physical_contests.iterrows()
    ):

        start = int(
            event["start_frame"]
        )


        end = int(
            event["end_frame"]
        )


        lock_end = (
            end
            +
            CONTEST_POST_LOCK_FRAMES
        )


        for frame in range(
            start,
            lock_end + 1,
        ):

            contest_lock_frames.add(
                frame
            )


            contest_source_map[
                frame
            ] = {
                "event_start":
                    start,

                "event_end":
                    end,

                "lock_end":
                    lock_end,

                "participants":
                    event.get(
                        "participants",
                        None,
                    ),
            }


    return (
        events,
        contest_lock_frames,
        contest_source_map,
    )


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

        (
            x1,
            y1,
            x2,
            y2,

        ) = box


        detections.append(
            {
                "bbox_x1":
                    float(x1),

                "bbox_y1":
                    float(y1),

                "bbox_x2":
                    float(x2),

                "bbox_y2":
                    float(y2),

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
                    float(y2),

                "confidence":
                    float(
                        confidence
                    ),
            }
        )


    return detections


# ============================================================
# Track helpers
# ============================================================

def get_track_row(
    tracks_by_frame,
    frame,
    track_id,
):

    if (
        frame
        not in
        tracks_by_frame
    ):

        return None


    if pd.isna(
        track_id
    ):

        return None


    frame_tracks = (
        tracks_by_frame[
            frame
        ]
    )


    matches = (
        frame_tracks[
            np.abs(
                frame_tracks[
                    "track_id"
                ]
                -
                float(track_id)
            )
            <
            0.1
        ]
    )


    if len(matches) == 0:

        return None


    return (
        matches.iloc[0]
    )


def match_controller_bbox(
    track_row,
    detections,
):

    if (
        track_row is None
        or
        len(detections) == 0
    ):

        return None


    track_x = float(
        track_row[
            "image_x"
        ]
    )


    track_y = float(
        track_row[
            "image_y"
        ]
    )


    best = None
    best_distance = None


    for detection in (
        detections
    ):

        distance = float(
            np.hypot(

                track_x
                -
                detection[
                    "bottom_x"
                ],

                track_y
                -
                detection[
                    "bottom_y"
                ],
            )
        )


        if (
            best_distance
            is None

            or

            distance
            <
            best_distance
        ):

            best = detection
            best_distance = distance


    if (
        best is None

        or

        best_distance
        >
        MAX_TRACK_BOX_MATCH_PX
    ):

        return None


    matched = (
        best.copy()
    )


    matched[
        "track_box_match_distance_px"
    ] = best_distance


    return matched


# ============================================================
# Geometry
# ============================================================

def calculate_geometry(
    ball_x,
    ball_y,
    bbox,
):

    if (
        bbox is None

        or

        pd.isna(
            ball_x
        )

        or

        pd.isna(
            ball_y
        )
    ):

        return {
            "verified":
                False,

            "bbox_height_px":
                np.nan,

            "foot_distance_px":
                np.nan,

            "foot_distance_norm":
                np.nan,

            "ball_rel_y":
                np.nan,
        }


    x1 = float(
        bbox[
            "bbox_x1"
        ]
    )

    y1 = float(
        bbox[
            "bbox_y1"
        ]
    )

    x2 = float(
        bbox[
            "bbox_x2"
        ]
    )

    y2 = float(
        bbox[
            "bbox_y2"
        ]
    )


    bbox_height = max(
        y2 - y1,
        1.0,
    )


    foot_x = (
        x1 + x2
    ) / 2.0


    foot_y = y2


    foot_distance_px = float(
        np.hypot(

            float(ball_x)
            -
            foot_x,

            float(ball_y)
            -
            foot_y,
        )
    )


    foot_distance_norm = (
        foot_distance_px
        /
        bbox_height
    )


    ball_rel_y = (

        (
            float(ball_y)
            -
            y1
        )

        /

        bbox_height
    )


    return {
        "verified":
            True,

        "bbox_height_px":
            float(
                bbox_height
            ),

        "foot_distance_px":
            float(
                foot_distance_px
            ),

        "foot_distance_norm":
            float(
                foot_distance_norm
            ),

        "ball_rel_y":
            float(
                ball_rel_y
            ),
    }


# ============================================================
# Controller Gate V2 decision
# ============================================================

def controller_gate_decision(
    frame,
    geometry,
    controller_run_length,
    contest_lock_frames,
):

    # Contest lock has highest priority.
    if (
        frame
        in
        contest_lock_frames
    ):

        return (
            "REJECT",
            "contest_lock",
            "contest",
        )


    if not geometry[
        "verified"
    ]:

        return (
            "UNVERIFIED",
            "no_reliable_bbox_geometry",
            "unknown",
        )


    distance_norm = (
        geometry[
            "foot_distance_norm"
        ]
    )


    distance_px = (
        geometry[
            "foot_distance_px"
        ]
    )


    bbox_height = (
        geometry[
            "bbox_height_px"
        ]
    )


    rel_y = (
        geometry[
            "ball_rel_y"
        ]
    )


    # Hard geometry veto.
    if (
        rel_y
        <
        HARD_MIN_BALL_REL_Y
    ):

        return (
            "REJECT",
            "ball_clearly_above_control_zone",
            "hard_geometry_veto",
        )


    if (
        rel_y
        >
        HARD_MAX_BALL_REL_Y
    ):

        return (
            "REJECT",
            "ball_clearly_below_control_zone",
            "hard_geometry_veto",
        )


    if (
        distance_norm
        >
        HARD_MAX_FOOT_DISTANCE_NORM
    ):

        return (
            "REJECT",
            "ball_extremely_far_from_controller",
            "hard_geometry_veto",
        )


    # Close control.
    if (

        distance_norm
        <=
        CLOSE_CONTROL_MAX_FOOT_DISTANCE_NORM

        and

        rel_y
        >=
        CLOSE_CONTROL_MIN_REL_Y

        and

        rel_y
        <=
        CLOSE_CONTROL_MAX_REL_Y
    ):

        return (
            "KEEP",
            "close_control_geometry",
            "close_control",
        )


    # Normal dribble continuity.
    if (

        distance_norm
        <=
        DRIBBLE_MAX_FOOT_DISTANCE_NORM

        and

        rel_y
        >=
        DRIBBLE_MIN_REL_Y

        and

        rel_y
        <=
        DRIBBLE_MAX_REL_Y

        and

        controller_run_length
        >=
        MIN_DRIBBLE_CONTROLLER_RUN
    ):

        return (
            "KEEP",
            "dribble_temporal_continuity",
            "dribble",
        )


    # Small/far-away player support.
    if (

        bbox_height
        <=
        SMALL_BBOX_HEIGHT_PX

        and

        distance_px
        <=
        SMALL_BBOX_MAX_FOOT_DISTANCE_PX

        and

        rel_y
        >=
        SMALL_BBOX_MIN_REL_Y

        and

        rel_y
        <=
        SMALL_BBOX_MAX_REL_Y

        and

        controller_run_length
        >=
        MIN_DRIBBLE_CONTROLLER_RUN
    ):

        return (
            "KEEP",
            "small_bbox_dribble_continuity",
            "dribble",
        )


    return (
        "AMBIGUOUS",
        "geometry_not_decisive",
        "ambiguous",
    )


# ============================================================
# Main gate pass
# ============================================================

def run_gate(
    video_path,
    model_path,
    tracks_by_frame,
    possession,
    possession_by_frame,
    controlled_frames,
    contest_lock_frames,
    contest_source_map,
    device,
):

    model = YOLO(
        str(model_path)
    )


    cap = cv2.VideoCapture(
        str(video_path)
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


    audit_rows = []

    frame_idx = 0


    print("")
    print(
        "=" * 58
    )
    print(
        "CONTROLLER GATE — PASS 1"
    )
    print(
        "Temporal + geometry audit"
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


        if (
            frame_idx
            not in
            controlled_frames
        ):

            frame_idx += 1
            continue


        row = (
            possession_by_frame.loc[
                frame_idx
            ]
        )


        if isinstance(
            row,
            pd.DataFrame,
        ):

            row = (
                row.iloc[0]
            )


        controller_team = (
            clean_team(
                row.get(
                    "controller_team_v5",
                    None,
                )
            )
        )


        controller_id = (
            clean_id(
                row.get(
                    "controller_id_v5",
                    np.nan,
                )
            )
        )


        controller_run_length = int(
            row.get(
                "controller_run_length",
                0,
            )
        )


        controller_run_position = int(
            row.get(
                "controller_run_position",
                0,
            )
        )


        ball_x = row.get(
            "ball_image_x",
            np.nan,
        )


        ball_y = row.get(
            "ball_image_y",
            np.nan,
        )


        detections = (
            detect_player_boxes(
                model,
                frame,
                device,
            )
        )


        track_row = (
            get_track_row(
                tracks_by_frame,
                frame_idx,
                controller_id,
            )
        )


        matched_bbox = (
            match_controller_bbox(
                track_row,
                detections,
            )
        )


        geometry = (
            calculate_geometry(
                ball_x,
                ball_y,
                matched_bbox,
            )
        )


        (
            gate_status,
            gate_reason,
            control_mode,

        ) = controller_gate_decision(

            frame_idx,

            geometry,

            controller_run_length,

            contest_lock_frames,
        )


        contest_info = (
            contest_source_map.get(
                frame_idx,
                {},
            )
        )


        audit_rows.append(
            {
                "frame":
                    frame_idx,

                "time_sec":
                    row.get(
                        "time_sec",
                        frame_idx / fps,
                    ),

                "possession_v5":
                    row.get(
                        "possession_v5",
                        None,
                    ),

                "ball_state_v5":
                    row.get(
                        "ball_state_v5",
                        None,
                    ),

                "controller_team":
                    controller_team,

                "controller_id":
                    controller_id,

                "controller_run_id":
                    row.get(
                        "controller_run_id",
                        np.nan,
                    ),

                "controller_run_length":
                    controller_run_length,

                "controller_run_position":
                    controller_run_position,

                "ball_speed_mps":
                    row.get(
                        "ball_speed_mps",
                        np.nan,
                    ),

                "ball_image_x":
                    ball_x,

                "ball_image_y":
                    ball_y,

                "contest_lock":
                    (
                        frame_idx
                        in
                        contest_lock_frames
                    ),

                "contest_participants":
                    contest_info.get(
                        "participants",
                        None,
                    ),

                "bbox_verified":
                    geometry[
                        "verified"
                    ],

                "bbox_x1":
                    (
                        matched_bbox[
                            "bbox_x1"
                        ]

                        if
                        matched_bbox
                        is not None

                        else
                        np.nan
                    ),

                "bbox_y1":
                    (
                        matched_bbox[
                            "bbox_y1"
                        ]

                        if
                        matched_bbox
                        is not None

                        else
                        np.nan
                    ),

                "bbox_x2":
                    (
                        matched_bbox[
                            "bbox_x2"
                        ]

                        if
                        matched_bbox
                        is not None

                        else
                        np.nan
                    ),

                "bbox_y2":
                    (
                        matched_bbox[
                            "bbox_y2"
                        ]

                        if
                        matched_bbox
                        is not None

                        else
                        np.nan
                    ),

                "bbox_height_px":
                    geometry[
                        "bbox_height_px"
                    ],

                "foot_distance_px":
                    geometry[
                        "foot_distance_px"
                    ],

                "foot_distance_norm":
                    geometry[
                        "foot_distance_norm"
                    ],

                "ball_rel_y":
                    geometry[
                        "ball_rel_y"
                    ],

                "gate_status":
                    gate_status,

                "gate_reason":
                    gate_reason,

                "control_mode":
                    control_mode,
            }
        )


        frame_idx += 1


    cap.release()


    audit = pd.DataFrame(
        audit_rows
    )


    audit = (
        audit
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )


    flagged = (
        audit[
            audit[
                "gate_status"
            ].isin(
                [
                    "REJECT",
                    "AMBIGUOUS",
                    "UNVERIFIED",
                ]
            )
        ]
        .copy()
    )


    return (
        audit,
        flagged,
        fps,
    )


# ============================================================
# Regression checks
# ============================================================

def run_regressions(
    audit,
):

    def get_frame(frame):

        rows = audit[
            audit[
                "frame"
            ]
            ==
            frame
        ]

        if len(rows) == 0:
            return None

        return rows.iloc[0]


    r175 = get_frame(
        REGRESSION_CONTEST_FRAME
    )


    reg175 = (

        r175 is not None

        and

        r175[
            "gate_status"
        ]
        ==
        "REJECT"

        and

        r175[
            "gate_reason"
        ]
        ==
        "contest_lock"
    )


    r226 = get_frame(
        REGRESSION_FALSE_CONTROL_FRAME
    )


    reg226 = (

        r226 is not None

        and

        r226[
            "gate_status"
        ]
        ==
        "REJECT"
    )


    r650 = get_frame(
        REGRESSION_DRIBBLE_FRAME
    )


    reg650 = (

        r650 is not None

        and

        r650[
            "gate_status"
        ]
        ==
        "KEEP"

        and

        r650[
            "control_mode"
        ]
        ==
        "dribble"
    )


    return {
        "frame_175_contest_reject":
            reg175,

        "frame_226_false_control_reject":
            reg226,

        "frame_650_dribble_keep":
            reg650,
    }


# ============================================================
# QA video
# ============================================================

def render_qa(
    video_path,
    audit,
    output_video,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot reopen video: "
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

        str(output_video),

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


    audit_by_frame = (
        audit
        .set_index(
            "frame"
        )
    )


    frame_idx = 0


    while True:

        ret, frame = (
            cap.read()
        )


        if not ret:
            break


        annotated = frame.copy()


        if (
            frame_idx
            in
            audit_by_frame.index
        ):

            result = (
                audit_by_frame.loc[
                    frame_idx
                ]
            )


            if isinstance(
                result,
                pd.DataFrame,
            ):

                result = (
                    result.iloc[0]
                )


            status = str(
                result[
                    "gate_status"
                ]
            )


            if status == "KEEP":

                color = (
                    0,
                    255,
                    0,
                )

            elif status == "REJECT":

                color = (
                    0,
                    0,
                    255,
                )

            elif status == "AMBIGUOUS":

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


            ball_x = (
                result[
                    "ball_image_x"
                ]
            )


            ball_y = (
                result[
                    "ball_image_y"
                ]
            )


            if (
                pd.notna(ball_x)
                and
                pd.notna(ball_y)
            ):

                cv2.circle(
                    annotated,

                    (
                        int(ball_x),
                        int(ball_y),
                    ),

                    10,

                    (
                        0,
                        255,
                        255,
                    ),

                    3,
                )


            if pd.notna(
                result[
                    "bbox_x1"
                ]
            ):

                x1 = int(
                    result[
                        "bbox_x1"
                    ]
                )

                y1 = int(
                    result[
                        "bbox_y1"
                    ]
                )

                x2 = int(
                    result[
                        "bbox_x2"
                    ]
                )

                y2 = int(
                    result[
                        "bbox_y2"
                    ]
                )


                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3,
                )


            lines = [
                (
                    f"CONTROLLER GATE | "
                    f"frame={frame_idx}"
                ),

                (
                    f"controller="
                    f"{result['controller_team']} "
                    f"#{int(result['controller_id'])}"
                ),

                (
                    f"{status} | "
                    f"{result['gate_reason']}"
                ),

                (
                    f"mode="
                    f"{result['control_mode']}"
                ),

                (
                    f"run="
                    f"{int(result['controller_run_position'])}/"
                    f"{int(result['controller_run_length'])}"
                ),
            ]


            overlay = (
                annotated.copy()
            )


            cv2.rectangle(
                overlay,
                (15, 15),
                (780, 190),
                (0, 0, 0),
                -1,
            )


            cv2.addWeighted(
                overlay,
                0.62,
                annotated,
                0.38,
                0,
                annotated,
            )


            y = 43


            for text in lines:

                cv2.putText(
                    annotated,
                    text,
                    (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                y += 30


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
            "Canonical Controller Gate V2 CLI."
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
        "--events",
        required=True,
    )


    parser.add_argument(
        "--output",
        required=True,
    )


    parser.add_argument(
        "--flagged-output",
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


    args = parser.parse_args()


    video_path = Path(
        args.video
    )


    model_path = Path(
        args.player_model
    )


    player_path = Path(
        args.players
    )


    possession_path = Path(
        args.possession
    )


    event_path = Path(
        args.events
    )


    output_path = Path(
        args.output
    )


    flagged_path = (

        Path(
            args.flagged_output
        )

        if args.flagged_output

        else

        output_path.parent
        /
        "controller_gate_flagged.csv"
    )


    require_file(
        video_path,
        "Input video",
    )


    require_file(
        model_path,
        "Player model",
    )


    require_file(
        player_path,
        "Player tracking CSV",
    )


    require_file(
        possession_path,
        "Possession V5.1 CSV",
    )


    require_file(
        event_path,
        "Air interaction events CSV",
    )


    ensure_parent(
        output_path
    )


    ensure_parent(
        flagged_path
    )


    if args.output_video:

        ensure_parent(
            Path(
                args.output_video
            )
        )


    device = resolve_device(
        args.device
    )


    print("")
    print(
        "=" * 60
    )
    print(
        "CONTROLLER GATE CLI"
    )
    print(
        "=" * 60
    )

    print(
        "Video:",
        video_path,
    )

    print(
        "Players:",
        player_path,
    )

    print(
        "Possession:",
        possession_path,
    )

    print(
        "Events:",
        event_path,
    )

    print(
        "Player model:",
        model_path,
    )

    print(
        "Device:",
        device,
    )


    tracks = pd.read_csv(
        player_path
    )


    possession = pd.read_csv(
        possession_path
    )


    events = pd.read_csv(
        event_path
    )


    (
        tracks,
        tracks_by_frame,

    ) = prepare_tracks(
        tracks
    )


    (
        possession,
        possession_by_frame,
        controlled_frames,
        run_groups,

    ) = prepare_possession(
        possession
    )


    (
        events,
        contest_lock_frames,
        contest_source_map,

    ) = prepare_contest_lock(
        events
    )


    print("")
    print(
        "Controlled frames to audit:",
        len(
            controlled_frames
        ),
    )

    print(
        "Controller runs:",
        len(
            run_groups
        ),
    )


    (
        audit,
        flagged,
        fps,

    ) = run_gate(

        video_path=(
            video_path
        ),

        model_path=(
            model_path
        ),

        tracks_by_frame=(
            tracks_by_frame
        ),

        possession=(
            possession
        ),

        possession_by_frame=(
            possession_by_frame
        ),

        controlled_frames=(
            controlled_frames
        ),

        contest_lock_frames=(
            contest_lock_frames
        ),

        contest_source_map=(
            contest_source_map
        ),

        device=device,
    )


    audit.to_csv(
        output_path,
        index=False,
    )


    flagged.to_csv(
        flagged_path,
        index=False,
    )


    checks = (
        run_regressions(
            audit
        )
    )


    print("")
    print(
        "=" * 60
    )
    print(
        "CONTROLLER GATE SUMMARY"
    )
    print(
        "=" * 60
    )


    print(
        "Controlled frames audited:",
        len(audit),
    )


    print("")
    print(
        "Gate status:"
    )


    print(
        audit[
            "gate_status"
        ]
        .value_counts()
        .to_string()
    )


    print("")
    print(
        "Control modes:"
    )


    print(
        audit[
            "control_mode"
        ]
        .value_counts()
        .to_string()
    )


    print("")
    print(
        "REJECT reasons:"
    )


    rejected = (
        audit[
            audit[
                "gate_status"
            ]
            ==
            "REJECT"
        ]
    )


    if len(rejected):

        print(
            rejected[
                "gate_reason"
            ]
            .value_counts()
            .to_string()
        )

    else:

        print(
            "No rejected frames."
        )


    print("")
    print(
        "AMBIGUOUS reasons:"
    )


    ambiguous = (
        audit[
            audit[
                "gate_status"
            ]
            ==
            "AMBIGUOUS"
        ]
    )


    if len(ambiguous):

        print(
            ambiguous[
                "gate_reason"
            ]
            .value_counts()
            .to_string()
        )

    else:

        print(
            "No ambiguous frames."
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


    if (
        args.output_video
        and
        not args.skip_qa
    ):

        print("")
        print(
            "Rendering QA video..."
        )


        render_qa(
            video_path,
            audit,
            Path(
                args.output_video
            ),
        )


        print(
            "QA video:",
            args.output_video,
        )


    print("")
    print(
        "=" * 60
    )
    print(
        "DONE"
    )
    print(
        "=" * 60
    )


    print(
        "FPS:",
        f"{fps:.3f}",
    )


    print(
        "Gate CSV:",
        output_path,
    )


    print(
        "Flagged CSV:",
        flagged_path,
    )


if __name__ == "__main__":

    main()
