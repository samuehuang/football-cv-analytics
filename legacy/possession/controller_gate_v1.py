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

OUTPUT_CSV = "outputs/controller_gate_v1.csv"
OUTPUT_FLAGGED_CSV = "outputs/controller_gate_v1_flagged.csv"
OUTPUT_VIDEO = "outputs/controller_gate_v1_qa.mp4"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "mps"


# ============================================================
# FOOTBALL PLAYER MODEL CLASSES
# ============================================================

GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2


# ============================================================
# PLAYER DETECTION
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
# A high-confidence physical AerialContest should not turn
# into Controlled possession immediately.
#
# Example:
# frame 172 = confirmed aerial duel
# frame 175 = still visibly contesting
#
# Extend lock four frames after the physical event.
# ============================================================

CONTEST_POST_LOCK_FRAMES = 4


# ============================================================
# CONTROL GEOMETRY GATE
#
# bbox height is used as player-scale normalization.
#
# A true controlled ball should normally be spatially
# consistent with the player's lower-body / foot region.
#
# This is a VETO only.
#
# It does NOT prove that a ball is airborne.
# ============================================================

MAX_CONTROL_FOOT_DISTANCE_NORM = 1.00

MIN_CONTROL_BALL_REL_Y = 0.45
MAX_CONTROL_BALL_REL_Y = 1.35


# ============================================================
# REGRESSION FRAMES
# ============================================================

# Still in aerial duel.
REGRESSION_CONTEST_FRAME = 175

# Ball visually still airborne / not controlled.
REGRESSION_AIRBORNE_FRAME = 226


# ============================================================
# HELPERS
# ============================================================

def clean_team(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"A", "B"}:
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


def safe_text(value, default="NONE"):

    if pd.isna(value):
        return default

    text = str(value).strip()

    if (
        text == ""
        or
        text.lower()
        in {
            "none",
            "nan",
        }
    ):
        return default

    return text


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
# LOAD DATA
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
# NORMALIZE TRACKS
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
# STABLE TEAM ASSIGNMENT
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
    subset=["frame"]
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
# NORMALIZE EVENTS
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
# BUILD HIGH-CONFIDENCE CONTEST LOCK
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


# ============================================================
# CONTROLLED FRAMES
# ============================================================

controlled = possession[
    possession["ball_state_v5"]
    ==
    "Controlled"
].copy()


controlled_frames = set(
    controlled["frame"]
    .astype(int)
    .tolist()
)


print("")
print(
    "Controlled frames to audit:",
    len(controlled_frames),
)


# ============================================================
# PLAYER DETECTOR
# ============================================================

model = YOLO(
    PLAYER_MODEL_PATH
)


def detect_player_boxes(frame):

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
                            x1 + x2
                        )
                        /
                        2.0
                    ),

                "bottom_y":
                    float(y2),

                "confidence":
                    float(confidence),
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
            float(track_id)
        )
        <
        0.1
    ]


    if len(matches) == 0:

        return None


    return matches.iloc[0]


# ============================================================
# MATCH CONTROLLER TRACK TO DETECTION BBOX
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
        track_row["image_x"]
    )

    track_y = float(
        track_row["image_y"]
    )


    best = None
    best_distance = None


    for detection in detections:

        distance = float(
            np.hypot(
                track_x
                -
                detection["bottom_x"],

                track_y
                -
                detection["bottom_y"],
            )
        )


        if (
            best_distance is None
            or
            distance < best_distance
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
    ] = best_distance


    return matched


# ============================================================
# CONTROL GEOMETRY
# ============================================================

def evaluate_control_geometry(
    ball_x,
    ball_y,
    bbox,
):

    if (
        pd.isna(ball_x)
        or
        pd.isna(ball_y)
        or
        bbox is None
    ):

        return {
            "verified":
                False,

            "geometry_pass":
                None,

            "reject_reason":
                "unverified_geometry",

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
        bbox["bbox_x1"]
    )

    y1 = float(
        bbox["bbox_y1"]
    )

    x2 = float(
        bbox["bbox_x2"]
    )

    y2 = float(
        bbox["bbox_y2"]
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


    foot_distance_ok = (

        foot_distance_norm
        <=
        MAX_CONTROL_FOOT_DISTANCE_NORM
    )


    vertical_ok = (

        ball_rel_y
        >=
        MIN_CONTROL_BALL_REL_Y

        and

        ball_rel_y
        <=
        MAX_CONTROL_BALL_REL_Y
    )


    geometry_pass = (

        foot_distance_ok
        and
        vertical_ok
    )


    if geometry_pass:

        reject_reason = None


    elif not foot_distance_ok:

        reject_reason = (
            "ball_too_far_from_control_feet"
        )


    elif not vertical_ok:

        if (
            ball_rel_y
            <
            MIN_CONTROL_BALL_REL_Y
        ):

            reject_reason = (
                "ball_above_lower_body_control_zone"
            )

        else:

            reject_reason = (
                "ball_outside_lower_body_control_zone"
            )


    else:

        reject_reason = (
            "control_geometry_failed"
        )


    return {
        "verified":
            True,

        "geometry_pass":
            bool(
                geometry_pass
            ),

        "reject_reason":
            reject_reason,

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
# PASS 1
#
# AUDIT ALL CONTROLLED FRAMES
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
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
    "CONTROLLER GATE V1 - PASS 1"
)

print(
    "Controlled-frame audit"
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

        row = row.iloc[0]


    controller_team = clean_team(
        row.get(
            "controller_team_v5",
            None,
        )
    )


    controller_id = clean_id(
        row.get(
            "controller_id_v5",
            np.nan,
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


    detections = detect_player_boxes(
        frame
    )


    track_row = get_track_row(
        frame_idx,
        controller_id,
    )


    matched_bbox = match_controller_bbox(
        track_row,
        detections,
    )


    geometry = evaluate_control_geometry(
        ball_x,
        ball_y,
        matched_bbox,
    )


    contest_lock = (

        frame_idx
        in
        contest_lock_frames
    )


    # ========================================================
    # DECISION
    # ========================================================

    if contest_lock:

        gate_status = (
            "REJECT"
        )

        gate_reason = (
            "contest_lock"
        )


    elif (

        geometry[
            "verified"
        ]

        and

        geometry[
            "geometry_pass"
        ]
        is False

    ):

        gate_status = (
            "REJECT"
        )

        gate_reason = (
            geometry[
                "reject_reason"
            ]
        )


    elif (

        geometry[
            "verified"
        ]

        and

        geometry[
            "geometry_pass"
        ]

    ):

        gate_status = (
            "KEEP"
        )

        gate_reason = (
            "control_geometry_pass"
        )


    else:

        # Insufficient evidence should not automatically
        # destroy an existing controller.
        gate_status = (
            "UNVERIFIED"
        )

        gate_reason = (
            "no_reliable_bbox_geometry"
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
                contest_lock,

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
                    if matched_bbox
                    is not None
                    else
                    np.nan
                ),

            "bbox_y1":
                (
                    matched_bbox[
                        "bbox_y1"
                    ]
                    if matched_bbox
                    is not None
                    else
                    np.nan
                ),

            "bbox_x2":
                (
                    matched_bbox[
                        "bbox_x2"
                    ]
                    if matched_bbox
                    is not None
                    else
                    np.nan
                ),

            "bbox_y2":
                (
                    matched_bbox[
                        "bbox_y2"
                    ]
                    if matched_bbox
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

            "geometry_pass":
                geometry[
                    "geometry_pass"
                ],

            "gate_status":
                gate_status,

            "gate_reason":
                gate_reason,
        }
    )


    frame_idx += 1


cap.release()


# ============================================================
# AUDIT DATAFRAME
# ============================================================

audit = pd.DataFrame(
    audit_rows
)


audit = (
    audit
    .sort_values("frame")
    .reset_index(drop=True)
)


audit.to_csv(
    OUTPUT_CSV,
    index=False,
)


flagged = audit[
    audit["gate_status"]
    ==
    "REJECT"
].copy()


flagged.to_csv(
    OUTPUT_FLAGGED_CSV,
    index=False,
)


# ============================================================
# REGRESSION
# ============================================================

def get_audit_frame(frame):

    matches = audit[
        audit["frame"]
        ==
        frame
    ]


    if len(matches) == 0:
        return None


    return matches.iloc[0]


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
    REGRESSION_AIRBORNE_FRAME
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


# ============================================================
# SUMMARY
# ============================================================

print("")
print(
    "========================================"
)

print(
    "CONTROLLER GATE V1 SUMMARY"
)

print(
    "========================================"
)

print("")

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
    "Reject reasons:"
)


if len(flagged) > 0:

    print(
        flagged[
            "gate_reason"
        ]
        .value_counts()
        .to_string()
    )

else:

    print(
        "No rejected controller frames."
    )


print("")

print(
    "REGRESSION CHECKS"
)


print(
    "frame 175 | "
    "aerial duel must reject controller:",
    (
        "PASS"
        if reg175
        else
        "FAIL"
    )
)


print(
    "frame 226 | "
    "airborne/not-controlled ball must reject controller:",
    (
        "PASS"
        if reg226
        else
        "FAIL"
    )
)


# ============================================================
# CONTROL GEOMETRY DISTRIBUTION
# ============================================================

verified = audit[
    audit["bbox_verified"]
    ==
    True
].copy()


if len(verified) > 0:

    print("")

    print(
        "FOOT DISTANCE NORM DISTRIBUTION"
    )


    print(
        verified[
            "foot_distance_norm"
        ]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(3)
        .to_string()
    )


# ============================================================
# REGRESSION DETAIL
# ============================================================

print("")

print(
    "REGRESSION DETAILS"
)


for regression_frame in [
    REGRESSION_CONTEST_FRAME,
    REGRESSION_AIRBORNE_FRAME,
]:

    result = get_audit_frame(
        regression_frame
    )


    print("")


    if result is None:

        print(
            f"frame {regression_frame}: "
            f"not found in Controlled frames"
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
        f"  speed="
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
        f"{result['gate_reason']}"
    )


# ============================================================
# PASS 2
#
# QA VIDEO
# ============================================================

audit_by_frame = (
    audit
    .set_index("frame")
)


cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
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
    "CONTROLLER GATE V1 - PASS 2"
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

        status_color = (
            0,
            255,
            0,
        )


    elif status == "REJECT":

        status_color = (
            0,
            0,
            255,
        )


    else:

        status_color = (
            0,
            165,
            255,
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
            scale=0.45,
            color=(
                0,
                255,
                255,
            ),
        )


    # ========================================================
    # CONTROLLER BBOX
    # ========================================================

    if (
        pd.notna(
            result[
                "bbox_x1"
            ]
        )
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
            status_color,
            3,
        )


        foot_x = int(
            (
                x1 + x2
            )
            /
            2
        )

        foot_y = y2


        cv2.circle(
            annotated,
            (
                foot_x,
                foot_y,
            ),
            7,
            status_color,
            -1,
        )


        if (
            pd.notna(ball_x)
            and
            pd.notna(ball_y)
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
                status_color,
                2,
            )


    # ========================================================
    # PANEL
    # ========================================================

    overlay = annotated.copy()


    cv2.rectangle(
        overlay,
        (
            15,
            15,
        ),
        (
            780,
            250,
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
            f"CONTROLLER GATE V1 | "
            f"frame={frame_idx} | "
            f"time={result['time_sec']:.2f}s"
        ),
        30,
        y,
        scale=0.58,
    )


    y += 31


    put_text(
        annotated,
        (
            f"V5.1: "
            f"{result['possession_v5']} / "
            f"{result['ball_state_v5']} / "
            f"{result['controller_team']} "
            f"#{int(result['controller_id'])}"
        ),
        30,
        y,
        scale=0.54,
    )


    y += 29


    put_text(
        annotated,
        (
            f"GATE: "
            f"{status} | "
            f"{result['gate_reason']}"
        ),
        30,
        y,
        scale=0.62,
        color=status_color,
    )


    y += 29


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


    y += 26


    if pd.notna(
        result[
            "foot_distance_norm"
        ]
    ):

        geometry_text = (
            f"Foot distance: "
            f"{result['foot_distance_px']:.1f}px | "
            f"bbox h={result['bbox_height_px']:.1f}px | "
            f"norm={result['foot_distance_norm']:.2f}"
        )

    else:

        geometry_text = (
            "Foot geometry: UNVERIFIED"
        )


    put_text(
        annotated,
        geometry_text,
        30,
        y,
        scale=0.46,
    )


    y += 26


    if pd.notna(
        result[
            "ball_rel_y"
        ]
    ):

        put_text(
            annotated,
            (
                f"Ball relative Y: "
                f"{result['ball_rel_y']:.2f} | "
                f"accepted="
                f"{MIN_CONTROL_BALL_REL_Y:.2f}"
                f".."
                f"{MAX_CONTROL_BALL_REL_Y:.2f}"
            ),
            30,
            y,
            scale=0.45,
        )


    # ========================================================
    # REJECT BANNER
    # ========================================================

    if status == "REJECT":

        cv2.rectangle(
            annotated,
            (
                15,
                270,
            ),
            (
                900,
                325,
            ),
            (
                0,
                0,
                255,
            ),
            -1,
        )


        put_text(
            annotated,
            (
                "CONTROL REJECTED | "
                "Controller evidence not trusted"
            ),
            30,
            305,
            scale=0.64,
            color=(
                255,
                255,
                255,
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
            "PASS | FRAME 175 AERIAL DUEL REJECTED"
            if reg175
            else
            "FAIL | FRAME 175 SHOULD REJECT CONTROLLER"
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
        REGRESSION_AIRBORNE_FRAME
    ):

        text = (
            "PASS | FRAME 226 CONTROL REJECTED"
            if reg226
            else
            "FAIL | FRAME 226 SHOULD REJECT CONTROLLER"
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


    writer.write(
        annotated
    )


    frame_idx += 1


cap.release()
writer.release()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("")
print(
    "Audit CSV:",
    OUTPUT_CSV,
)

print(
    "Flagged CSV:",
    OUTPUT_FLAGGED_CSV,
)

print(
    "QA Video:",
    OUTPUT_VIDEO,
)
