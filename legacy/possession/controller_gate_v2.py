import os

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = "videos/match.mp4"

PLAYER_MODEL_PATH = "models/football-player-detection.pt"

PLAYER_TRACK_CSV = "outputs/tracking_data_clean_v2.csv"
POSSESSION_CSV = "outputs/possession_v5_1_frames.csv"
AIR_TOUCH_EVENT_CSV = "outputs/air_touch_v4_events.csv"

OUTPUT_CSV = "outputs/controller_gate_v2.csv"
OUTPUT_FLAGGED_CSV = "outputs/controller_gate_v2_flagged.csv"
OUTPUT_VIDEO = "outputs/controller_gate_v2_qa.mp4"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "mps"


# ============================================================
# PLAYER MODEL CLASSES
# ============================================================

GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2


# ============================================================
# DETECTION
# ============================================================

PLAYER_CONF = 0.25
PLAYER_IMGSZ = 1280


# ============================================================
# STABLE TEAM
# ============================================================

MIN_DOMINANT_TEAM_SHARE = 0.70


# ============================================================
# TRACK <-> BBOX MATCH
# ============================================================

MAX_TRACK_BOX_MATCH_PX = 55.0


# ============================================================
# CONTEST LOCK
#
# Known physical aerial duel:
#
# frame 172
# A #1 + B #14
#
# Visual QA showed frame 175 was STILL contesting.
#
# Therefore prevent controller creation for a short period
# after the confirmed physical aerial contest.
# ============================================================

CONTEST_POST_LOCK_FRAMES = 4


# ============================================================
# HARD GEOMETRY VETO
#
# These are intentionally conservative.
#
# We only REJECT when geometry is clearly implausible.
# ============================================================

# Ball very far from controller relative to bbox height.
HARD_MAX_FOOT_DISTANCE_NORM = 1.35


# Ball extremely high relative to bbox.
#
# 0.0 = bbox top
# 1.0 = bbox bottom / feet
HARD_MIN_BALL_REL_Y = 0.30


# Ball well below expected feet region.
HARD_MAX_BALL_REL_Y = 1.45


# ============================================================
# CLOSE CONTROL
# ============================================================

CLOSE_CONTROL_MAX_FOOT_DISTANCE_NORM = 1.00

CLOSE_CONTROL_MIN_REL_Y = 0.50
CLOSE_CONTROL_MAX_REL_Y = 1.35


# ============================================================
# DRIBBLE CONTINUITY
#
# A player can push the ball forward while still controlling
# the dribble.
#
# So distance > 1 bbox height is NOT automatically a failure.
# ============================================================

DRIBBLE_MAX_FOOT_DISTANCE_NORM = 1.20

DRIBBLE_MIN_REL_Y = 0.70
DRIBBLE_MAX_REL_Y = 1.20

MIN_DRIBBLE_CONTROLLER_RUN = 3


# ============================================================
# SMALL / FAR-AWAY PLAYER SUPPORT
#
# When the bbox is tiny, normalized distance becomes noisy.
#
# Example:
# frame 650
# bbox h ~= 31 px
# foot distance ~= 31 px
# norm ~= 1.01
#
# Visually this is valid dribbling.
# ============================================================

SMALL_BBOX_HEIGHT_PX = 35.0

SMALL_BBOX_MAX_FOOT_DISTANCE_PX = 38.0

SMALL_BBOX_MIN_REL_Y = 0.70
SMALL_BBOX_MAX_REL_Y = 1.20


# ============================================================
# REGRESSION FRAMES
# ============================================================

# Aerial duel still active.
REGRESSION_CONTEST_FRAME = 175

# Ball visually still airborne / not controlled.
REGRESSION_FALSE_CONTROL_FRAME = 226

# Valid dribble where ball is pushed ahead.
REGRESSION_DRIBBLE_FRAME = 650


# ============================================================
# HELPERS
# ============================================================

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


def put_text(
    image,
    text,
    x,
    y,
    scale=0.5,
    color=(255, 255, 255),
    thickness=2,
):

    cv2.putText(
        image,
        str(text),
        (
            int(x),
            int(y),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# LOAD
# ============================================================

print("")
print("Loading player tracking...")

tracks = pd.read_csv(
    PLAYER_TRACK_CSV
)


print("Loading Possession V5.1...")

possession = pd.read_csv(
    POSSESSION_CSV
)


print("Loading AirTouch V4 events...")

events = pd.read_csv(
    AIR_TOUCH_EVENT_CSV
)


# ============================================================
# NORMALIZE TRACKING
# ============================================================

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


tracks = tracks.dropna(
    subset=[
        "frame",
        "track_id",
        "image_x",
        "image_y",
    ]
).copy()


tracks["frame"] = (
    tracks["frame"]
    .astype(int)
)


tracks = tracks[
    tracks["team"].isin(
        [
            "A",
            "B",
        ]
    )
].copy()


# ============================================================
# STABLE TEAM
# ============================================================

stable_team_map = {}


for track_id, group in tracks.groupby(
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


tracks = tracks[
    tracks[
        "stable_team"
    ].notna()
].copy()


# Do not relabel contaminated rows.
tracks = tracks[
    tracks["team"]
    ==
    tracks["stable_team"]
].copy()


tracks_by_frame = {

    int(frame):
        group.copy()

    for frame, group
    in tracks.groupby("frame")
}


# ============================================================
# NORMALIZE POSSESSION
# ============================================================

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


possession = possession.dropna(
    subset=[
        "frame"
    ]
).copy()


possession["frame"] = (
    possession["frame"]
    .astype(int)
)


possession = (
    possession
    .sort_values("frame")
    .reset_index(drop=True)
)


possession_by_frame = (
    possession
    .set_index("frame")
)


# ============================================================
# NORMALIZE AIR TOUCH EVENTS
# ============================================================

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


events = events.dropna(
    subset=[
        "start_frame",
        "end_frame",
        "final_class",
    ]
).copy()


events["start_frame"] = (
    events["start_frame"]
    .astype(int)
)


events["end_frame"] = (
    events["end_frame"]
    .astype(int)
)


# ============================================================
# CONTEST LOCK
# ============================================================

contest_lock_frames = set()

contest_source_map = {}


physical_contests = events[
    events["final_class"]
    ==
    "AerialContest"
].copy()


for _, event in physical_contests.iterrows():

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


# ============================================================
# BUILD CONTROLLER RUNS
#
# This is crucial for dribbling.
#
# Consecutive Controlled frames belonging to the same
# controller are grouped into one temporal run.
# ============================================================

possession["controller_run_id"] = np.nan
possession["controller_run_length"] = 0
possession["controller_run_position"] = 0


run_groups = []

current_indices = []

current_team = None
current_id = np.nan
previous_frame = None


def flush_run():

    global current_indices
    global current_team
    global current_id
    global previous_frame

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


for index, row in possession.iterrows():

    frame = int(
        row[
            "frame"
        ]
    )


    state = str(
        row.get(
            "ball_state_v5",
            ""
        )
    )


    team = clean_team(
        row.get(
            "controller_team_v5",
            None
        )
    )


    track_id = clean_id(
        row.get(
            "controller_id_v5",
            np.nan
        )
    )


    valid_control = (

        state
        ==
        "Controlled"

        and

        team
        in
        {
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


    if len(current_indices) == 0:

        current_indices = [
            index
        ]

        current_team = (
            team
        )

        current_id = (
            track_id
        )

        previous_frame = (
            frame
        )

        continue


    same_controller = (

        team
        ==
        current_team

        and

        same_player_id(
            track_id,
            current_id
        )
    )


    consecutive = (

        previous_frame
        is not None

        and

        frame
        ==
        previous_frame
        +
        1
    )


    if (
        same_controller
        and
        consecutive
    ):

        current_indices.append(
            index
        )

        previous_frame = (
            frame
        )


    else:

        flush_run()

        current_indices = [
            index
        ]

        current_team = (
            team
        )

        current_id = (
            track_id
        )

        previous_frame = (
            frame
        )


flush_run()


# ============================================================
# WRITE RUN METADATA
# ============================================================

for run_id, run in enumerate(
    run_groups,
    start=1,
):

    indices = run[
        "indices"
    ]


    run_length = len(
        indices
    )


    for position, index in enumerate(
        indices,
        start=1,
    ):

        possession.loc[
            index,
            "controller_run_id"
        ] = run_id


        possession.loc[
            index,
            "controller_run_length"
        ] = run_length


        possession.loc[
            index,
            "controller_run_position"
        ] = position


# Refresh frame lookup after adding run fields.
possession_by_frame = (
    possession
    .set_index(
        "frame"
    )
)


controlled = possession[
    possession[
        "ball_state_v5"
    ]
    ==
    "Controlled"
].copy()


controlled_frames = set(
    controlled[
        "frame"
    ]
    .astype(int)
    .tolist()
)


print("")
print(
    "Controlled frames to audit:",
    len(
        controlled_frames
    )
)

print(
    "Controller runs:",
    len(
        run_groups
    )
)


# ============================================================
# PLAYER MODEL
# ============================================================

model = YOLO(
    PLAYER_MODEL_PATH
)


def detect_player_boxes(
    frame
):

    result = model(

        frame,

        imgsz=PLAYER_IMGSZ,

        conf=PLAYER_CONF,

        classes=[
            GOALKEEPER_CLASS,
            PLAYER_CLASS,
        ],

        device=DEVICE,

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


    for box, confidence in zip(
        xyxy,
        confidences,
    ):

        x1, y1, x2, y2 = box


        detections.append(
            {
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

                "bottom_x":
                    float(
                        (
                            x1 + x2
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
# TRACK ROW
# ============================================================

def get_track_row(
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


    matches = frame_tracks[
        np.abs(
            frame_tracks[
                "track_id"
            ]
            -
            float(
                track_id
            )
        )
        <
        0.1
    ]


    if len(matches) == 0:

        return None


    return matches.iloc[0]


# ============================================================
# TRACK <-> BBOX
# ============================================================

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


    for detection in detections:

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
            best_distance is None

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


    matched = best.copy()


    matched[
        "track_box_match_distance_px"
    ] = (
        best_distance
    )


    return matched


# ============================================================
# GEOMETRY
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


    foot_y = (
        y2
    )


    foot_distance_px = float(
        np.hypot(

            float(
                ball_x
            )
            -
            foot_x,

            float(
                ball_y
            )
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
            float(
                ball_y
            )
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
# V2 GATE DECISION
# ============================================================

def controller_gate_decision(
    frame,
    geometry,
    controller_run_length,
):

    # ========================================================
    # 1. CONTEST LOCK
    # ========================================================

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


    # ========================================================
    # 2. UNVERIFIED
    # ========================================================

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


    # ========================================================
    # 3. HARD VETO
    # ========================================================

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


    # ========================================================
    # 4. CLOSE CONTROL
    # ========================================================

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


    # ========================================================
    # 5. NORMAL DRIBBLE CONTINUITY
    # ========================================================

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


    # ========================================================
    # 6. SMALL BBOX / FAR-AWAY DRIBBLE
    #
    # Raw px distance becomes useful because normalized
    # distance is noisy when bbox height is tiny.
    # ========================================================

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


    # ========================================================
    # 7. AMBIGUOUS
    #
    # V2 deliberately avoids destroying a controller when
    # evidence is neither clearly good nor clearly bad.
    # ========================================================

    return (
        "AMBIGUOUS",
        "geometry_not_decisive",
        "ambiguous",
    )


# ============================================================
# PASS 1
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: "
        f"{VIDEO_PATH}"
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
    "========================================"
)

print(
    "CONTROLLER GATE V2 - PASS 1"
)

print(
    "Temporal + geometry audit"
)

print(
    "========================================"
)

print("")


while True:

    ret, frame = cap.read()


    if not ret:
        break


    if (
        frame_idx
        not in
        controlled_frames
    ):

        frame_idx += 1
        continue


    row = possession_by_frame.loc[
        frame_idx
    ]


    if isinstance(
        row,
        pd.DataFrame
    ):

        row = (
            row.iloc[0]
        )


    controller_team = clean_team(
        row.get(
            "controller_team_v5",
            None
        )
    )


    controller_id = clean_id(
        row.get(
            "controller_id_v5",
            np.nan
        )
    )


    controller_run_length = int(
        row.get(
            "controller_run_length",
            0
        )
    )


    controller_run_position = int(
        row.get(
            "controller_run_position",
            0
        )
    )


    ball_x = row.get(
        "ball_image_x",
        np.nan
    )


    ball_y = row.get(
        "ball_image_y",
        np.nan
    )


    detections = (
        detect_player_boxes(
            frame
        )
    )


    track_row = get_track_row(
        frame_idx,
        controller_id
    )


    matched_bbox = (
        match_controller_bbox(
            track_row,
            detections
        )
    )


    geometry = (
        calculate_geometry(
            ball_x,
            ball_y,
            matched_bbox
        )
    )


    (
        gate_status,
        gate_reason,
        control_mode,

    ) = controller_gate_decision(

        frame_idx,

        geometry,

        controller_run_length
    )


    contest_info = (
        contest_source_map.get(
            frame_idx,
            {}
        )
    )


    audit_rows.append(
        {
            "frame":
                frame_idx,

            "time_sec":
                row.get(
                    "time_sec",
                    frame_idx / fps
                ),

            "possession_v5":
                row.get(
                    "possession_v5",
                    None
                ),

            "ball_state_v5":
                row.get(
                    "ball_state_v5",
                    None
                ),

            "controller_team":
                controller_team,

            "controller_id":
                controller_id,

            "controller_run_id":
                row.get(
                    "controller_run_id",
                    np.nan
                ),

            "controller_run_length":
                controller_run_length,

            "controller_run_position":
                controller_run_position,

            "ball_speed_mps":
                row.get(
                    "ball_speed_mps",
                    np.nan
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
                    None
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

                    if matched_bbox
                    is not None

                    else np.nan
                ),

            "bbox_y1":
                (
                    matched_bbox[
                        "bbox_y1"
                    ]

                    if matched_bbox
                    is not None

                    else np.nan
                ),

            "bbox_x2":
                (
                    matched_bbox[
                        "bbox_x2"
                    ]

                    if matched_bbox
                    is not None

                    else np.nan
                ),

            "bbox_y2":
                (
                    matched_bbox[
                        "bbox_y2"
                    ]

                    if matched_bbox
                    is not None

                    else np.nan
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


# ============================================================
# RESULTS
# ============================================================

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


audit.to_csv(
    OUTPUT_CSV,
    index=False
)


flagged = audit[
    audit[
        "gate_status"
    ].isin(
        [
            "REJECT",
            "AMBIGUOUS",
            "UNVERIFIED",
        ]
    )
].copy()


flagged.to_csv(
    OUTPUT_FLAGGED_CSV,
    index=False
)


# ============================================================
# REGRESSION
# ============================================================

def get_audit_frame(
    frame
):

    rows = audit[
        audit["frame"]
        ==
        frame
    ]


    if len(rows) == 0:
        return None


    return rows.iloc[0]


r175 = get_audit_frame(
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


r226 = get_audit_frame(
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


r650 = get_audit_frame(
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


# ============================================================
# SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "CONTROLLER GATE V2 SUMMARY"
)

print(
    "========================================"
)

print("")

print(
    "Controlled frames audited:",
    len(
        audit
    )
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


rejected = audit[
    audit[
        "gate_status"
    ]
    ==
    "REJECT"
]


if len(rejected) > 0:

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


ambiguous = audit[
    audit[
        "gate_status"
    ]
    ==
    "AMBIGUOUS"
]


if len(ambiguous) > 0:

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


# ============================================================
# REGRESSION SUMMARY
# ============================================================

print("")

print(
    "REGRESSION CHECKS"
)


print(
    "frame 175 | "
    "aerial duel -> REJECT:",
    (
        "PASS"
        if reg175
        else
        "FAIL"
    )
)


print(
    "frame 226 | "
    "airborne/not-controlled -> REJECT:",
    (
        "PASS"
        if reg226
        else
        "FAIL"
    )
)


print(
    "frame 650 | "
    "valid dribble -> KEEP:",
    (
        "PASS"
        if reg650
        else
        "FAIL"
    )
)


# ============================================================
# REGRESSION DETAILS
# ============================================================

print("")

print(
    "REGRESSION DETAILS"
)


for frame in [
    REGRESSION_CONTEST_FRAME,
    REGRESSION_FALSE_CONTROL_FRAME,
    REGRESSION_DRIBBLE_FRAME,
]:

    result = get_audit_frame(
        frame
    )


    print("")


    if result is None:

        print(
            f"frame {frame}: "
            f"not found"
        )

        continue


    print(
        f"frame={int(result['frame'])} | "
        f"time={result['time_sec']:.2f}s | "
        f"controller="
        f"{result['controller_team']} "
        f"#{int(result['controller_id'])}"
    )


    print(
        f"  run="
        f"{int(result['controller_run_position'])}"
        f"/"
        f"{int(result['controller_run_length'])}"
    )


    print(
        f"  ball_speed="
        f"{result['ball_speed_mps']:.2f} m/s"
    )


    print(
        f"  contest_lock="
        f"{result['contest_lock']}"
    )


    print(
        f"  foot_distance_px="
        f"{result['foot_distance_px']:.2f}"
    )


    print(
        f"  bbox_height_px="
        f"{result['bbox_height_px']:.2f}"
    )


    print(
        f"  foot_distance_norm="
        f"{result['foot_distance_norm']:.3f}"
    )


    print(
        f"  ball_rel_y="
        f"{result['ball_rel_y']:.3f}"
    )


    print(
        f"  RESULT="
        f"{result['gate_status']} | "
        f"{result['gate_reason']} | "
        f"{result['control_mode']}"
    )


# ============================================================
# FLAGGED SEGMENTS
# ============================================================

print("")

print(
    "========================================"
)

print(
    "FLAGGED SEGMENTS"
)

print(
    "========================================"
)

print("")


flagged_for_segments = audit[
    audit[
        "gate_status"
    ]
    !=
    "KEEP"
].copy()


segments = []


if len(
    flagged_for_segments
) > 0:

    flagged_for_segments = (
        flagged_for_segments
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )


    current_start = None
    current_end = None
    statuses = set()
    reasons = set()


    for _, row in (
        flagged_for_segments.iterrows()
    ):

        frame = int(
            row[
                "frame"
            ]
        )


        if current_start is None:

            current_start = (
                frame
            )

            current_end = (
                frame
            )

            statuses = {
                str(
                    row[
                        "gate_status"
                    ]
                )
            }

            reasons = {
                str(
                    row[
                        "gate_reason"
                    ]
                )
            }

            continue


        if (
            frame
            <=
            current_end
            +
            1
        ):

            current_end = (
                frame
            )

            statuses.add(
                str(
                    row[
                        "gate_status"
                    ]
                )
            )

            reasons.add(
                str(
                    row[
                        "gate_reason"
                    ]
                )
            )


        else:

            segments.append(
                (
                    current_start,
                    current_end,
                    sorted(
                        statuses
                    ),
                    sorted(
                        reasons
                    ),
                )
            )


            current_start = (
                frame
            )

            current_end = (
                frame
            )

            statuses = {
                str(
                    row[
                        "gate_status"
                    ]
                )
            }

            reasons = {
                str(
                    row[
                        "gate_reason"
                    ]
                )
            }


    if current_start is not None:

        segments.append(
            (
                current_start,
                current_end,
                sorted(
                    statuses
                ),
                sorted(
                    reasons
                ),
            )
        )


for (
    start,
    end,
    statuses,
    reasons,

) in segments:

    print(
        f"{start:4d}-{end:4d} | "
        f"{start / fps:.2f}-"
        f"{end / fps:.2f}s | "
        f"{end - start + 1:2d} frames | "
        f"{','.join(statuses)} | "
        f"{', '.join(reasons)}"
    )


# ============================================================
# QA VIDEO
# ============================================================

audit_by_frame = (
    audit
    .set_index(
        "frame"
    )
)


cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot reopen video: "
        f"{VIDEO_PATH}"
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
        f"Cannot create QA video: "
        f"{OUTPUT_VIDEO}"
    )


frame_idx = 0


print("")
print(
    "========================================"
)

print(
    "CONTROLLER GATE V2 - PASS 2"
)

print(
    "QA video"
)

print(
    "========================================"
)

print("")


while True:

    ret, frame = cap.read()


    if not ret:
        break


    annotated = frame.copy()


    if (
        frame_idx
        not in
        audit_by_frame.index
    ):

        writer.write(
            annotated
        )

        frame_idx += 1

        continue


    result = audit_by_frame.loc[
        frame_idx
    ]


    if isinstance(
        result,
        pd.DataFrame
    ):

        result = result.iloc[0]


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


    # ========================================================
    # BALL
    # ========================================================

    ball_x = result[
        "ball_image_x"
    ]

    ball_y = result[
        "ball_image_y"
    ]


    if (
        pd.notna(
            ball_x
        )
        and
        pd.notna(
            ball_y
        )
    ):

        cv2.circle(
            annotated,

            (
                int(ball_x),
                int(ball_y),
            ),

            11,

            (
                0,
                255,
                255,
            ),

            3,
        )


        put_text(
            annotated,

            "BALL",

            int(ball_x) + 14,
            int(ball_y) - 10,

            scale=0.44,

            color=(
                0,
                255,
                255,
            ),
        )


    # ========================================================
    # BBOX
    # ========================================================

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


        foot_x = int(
            (
                x1 + x2
            )
            /
            2
        )

        foot_y = (
            y2
        )


        cv2.circle(
            annotated,

            (
                foot_x,
                foot_y,
            ),

            6,

            color,

            -1,
        )


        if (
            pd.notna(
                ball_x
            )
            and
            pd.notna(
                ball_y
            )
        ):

            cv2.line(
                annotated,

                (
                    foot_x,
                    foot_y,
                ),

                (
                    int(ball_x),
                    int(ball_y),
                ),

                color,

                2,
            )


    # ========================================================
    # PANEL
    # ========================================================

    overlay = (
        annotated.copy()
    )


    cv2.rectangle(
        overlay,

        (
            15,
            15,
        ),

        (
            820,
            280,
        ),

        (
            0,
            0,
            0,
        ),

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


    put_text(
        annotated,

        (
            f"CONTROLLER GATE V2 | "
            f"frame={frame_idx} | "
            f"time={result['time_sec']:.2f}s"
        ),

        30,
        y,

        scale=0.58,
    )


    y += 30


    put_text(
        annotated,

        (
            f"V5.1 controller: "
            f"{result['controller_team']} "
            f"#{int(result['controller_id'])}"
        ),

        30,
        y,

        scale=0.52,
    )


    y += 28


    put_text(
        annotated,

        (
            f"GATE: "
            f"{status} | "
            f"{result['gate_reason']}"
        ),

        30,
        y,

        scale=0.60,

        color=color,
    )


    y += 28


    put_text(
        annotated,

        (
            f"MODE: "
            f"{result['control_mode']}"
        ),

        30,
        y,

        scale=0.50,

        color=color,
    )


    y += 27


    put_text(
        annotated,

        (
            f"Controller run: "
            f"{int(result['controller_run_position'])}/"
            f"{int(result['controller_run_length'])}"
        ),

        30,
        y,

        scale=0.48,
    )


    y += 27


    put_text(
        annotated,

        (
            f"Contest lock: "
            f"{bool(result['contest_lock'])}"
        ),

        30,
        y,

        scale=0.48,
    )


    y += 27


    if pd.notna(
        result[
            "foot_distance_norm"
        ]
    ):

        put_text(
            annotated,

            (
                f"Foot dist="
                f"{result['foot_distance_px']:.1f}px | "
                f"bbox h="
                f"{result['bbox_height_px']:.1f}px | "
                f"norm="
                f"{result['foot_distance_norm']:.2f}"
            ),

            30,
            y,

            scale=0.45,
        )


        y += 25


        put_text(
            annotated,

            (
                f"Ball rel-Y="
                f"{result['ball_rel_y']:.2f} | "
                f"speed="
                f"{result['ball_speed_mps']:.2f}m/s"
            ),

            30,
            y,

            scale=0.45,
        )


    # ========================================================
    # STATUS BANNER
    # ========================================================

    cv2.rectangle(
        annotated,

        (
            15,
            300,
        ),

        (
            950,
            350,
        ),

        color,

        -1,
    )


    if status == "KEEP":

        banner = (
            "CONTROL KEPT | "
            "Controller evidence accepted"
        )


    elif status == "REJECT":

        banner = (
            "CONTROL REJECTED | "
            "Controller evidence not trusted"
        )


    elif status == "AMBIGUOUS":

        banner = (
            "CONTROL AMBIGUOUS | "
            "Do not override automatically"
        )


    else:

        banner = (
            "CONTROL UNVERIFIED | "
            "Insufficient geometry"
        )


    put_text(
        annotated,

        banner,

        30,
        333,

        scale=0.60,

        color=(
            0,
            0,
            0,
        ),
    )


    # ========================================================
    # REGRESSION BANNERS
    # ========================================================

    if (
        frame_idx
        ==
        REGRESSION_CONTEST_FRAME
    ):

        text = (

            "PASS | FRAME 175 CONTEST REJECTED"

            if reg175

            else

            "FAIL | FRAME 175"
        )


        put_text(
            annotated,

            text,

            30,
            height - 55,

            scale=0.65,

            color=(
                0,
                255,
                0,
            )
            if reg175
            else
            (
                0,
                0,
                255,
            ),
        )


    if (
        frame_idx
        ==
        REGRESSION_FALSE_CONTROL_FRAME
    ):

        text = (

            "PASS | FRAME 226 FALSE CONTROL REJECTED"

            if reg226

            else

            "FAIL | FRAME 226"
        )


        put_text(
            annotated,

            text,

            30,
            height - 55,

            scale=0.65,

            color=(
                0,
                255,
                0,
            )
            if reg226
            else
            (
                0,
                0,
                255,
            ),
        )


    if (
        frame_idx
        ==
        REGRESSION_DRIBBLE_FRAME
    ):

        text = (

            "PASS | FRAME 650 DRIBBLE KEPT"

            if reg650

            else

            "FAIL | FRAME 650 SHOULD REMAIN CONTROLLED"
        )


        put_text(
            annotated,

            text,

            30,
            height - 55,

            scale=0.65,

            color=(
                0,
                255,
                0,
            )
            if reg650
            else
            (
                0,
                0,
                255,
            ),
        )


    writer.write(
        annotated
    )


    frame_idx += 1


cap.release()
writer.release()


# ============================================================
# OUTPUT
# ============================================================

print("")

print(
    "Audit CSV:",
    OUTPUT_CSV
)

print(
    "Flagged CSV:",
    OUTPUT_FLAGGED_CSV
)

print(
    "QA Video:",
    OUTPUT_VIDEO
)
