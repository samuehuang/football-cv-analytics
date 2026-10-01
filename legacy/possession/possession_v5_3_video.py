import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = "videos/match.mp4"

POSSESSION_CSV = "outputs/possession_v5_3_frames.csv"
TRACKING_CSV = "outputs/tracking_data_clean_v2.csv"

OUTPUT_VIDEO = "outputs/possession_v5_3_qa.mp4"
OUTPUT_KEY_FRAMES = "outputs/possession_v5_3_key_frames.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# DISPLAY
# ============================================================

PANEL_X = 15
PANEL_Y = 15
PANEL_W = 900
PANEL_H = 325

PLAYER_RADIUS = 6
CONTROLLER_RADIUS = 15
LAST_TOUCH_RADIUS = 12
BALL_RADIUS = 10


# BGR
COLOR_A = (255, 80, 40)
COLOR_B = (40, 40, 255)

COLOR_BALL = (0, 255, 255)

COLOR_CONTROLLER = (0, 255, 0)
COLOR_LAST_TOUCH = (255, 0, 255)

COLOR_CONTEST = (0, 215, 255)
COLOR_UNCONTROLLED = (0, 100, 255)
COLOR_UNKNOWN = (170, 170, 170)

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


# ============================================================
# KEY QA FRAMES
# ============================================================

KEY_FRAMES = {
    159: "B long ball / InTransit",
    172: "Physical AerialContest starts",
    173: "ContestCarry",
    174: "ContestCarry",
    175: "False controller must stay blocked",
    176: "ContestCarry / controller NONE",
    177: "Contest resolution -> A InTransit",

    203: "A6 reception/bounce regression",

    213: "Hard controller rejection",

    225: "False control rejection",
    226: "Airborne / not controlled",

    232: "Hard controller rejection",
    233: "Hard controller rejection",
    234: "Hard controller rejection",
    235: "Hard controller rejection",

    253: "AerialContest regression",

    281: "B10 AirTouch",
    287: "Post-contest fallback",

    335: "Upper-body geometry rejection",
    336: "Upper-body geometry rejection",
    337: "Upper-body geometry rejection",
    338: "Upper-body geometry rejection",
    339: "Upper-body geometry rejection",

    650: "Valid A11 dribble",
}


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


def clean_text(value, default=None):

    if pd.isna(value):
        return default

    text = str(value).strip()

    if (
        text == ""
        or
        text.lower() in {
            "none",
            "nan",
        }
    ):
        return default

    return text


def clean_id(value):

    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def as_bool(value):

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return value in {
        "true",
        "1",
        "yes",
    }


def format_player(team, track_id):

    team = clean_team(team)

    if (
        team is None
        or
        pd.isna(track_id)
    ):
        return "NONE"

    return (
        f"{team} "
        f"#{int(float(track_id))}"
    )


def put_text(
    image,
    text,
    x,
    y,
    scale=0.48,
    color=COLOR_WHITE,
    thickness=2,
):

    cv2.putText(
        image,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_transparent_panel(
    image,
    x1,
    y1,
    x2,
    y2,
    alpha=0.65,
):

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        COLOR_BLACK,
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
        image,
    )


# ============================================================
# LOAD POSSESSION
# ============================================================

print("")
print("Loading Possession V5.3...")

possession = pd.read_csv(
    POSSESSION_CSV
)


numeric_columns = [
    "frame",
    "time_sec",

    "ball_image_x",
    "ball_image_y",
    "ball_speed_mps",

    "controller_id_v5_2",
    "controller_id_v5_3",

    "last_touch_id_v5_2",
    "last_touch_id_v5_3",
]


for column in numeric_columns:

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
# LOAD PLAYER TRACKING
# ============================================================

print("Loading player tracking...")

tracks = pd.read_csv(
    TRACKING_CSV
)


for column in [
    "frame",
    "track_id",
    "image_x",
    "image_y",
]:

    if column in tracks.columns:

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


# ============================================================
# STABLE TEAM MAP
# ============================================================

stable_team_map = {}


valid_team_tracks = tracks[
    tracks["team"].isin(
        ["A", "B"]
    )
].copy()


for track_id, group in (
    valid_team_tracks.groupby(
        "track_id"
    )
):

    counts = (
        group["team"]
        .value_counts()
    )

    if len(counts) == 0:
        continue

    stable_team_map[
        float(track_id)
    ] = counts.index[0]


tracks["stable_team"] = (
    tracks["track_id"]
    .map(stable_team_map)
)


tracks_by_frame = {
    int(frame): group.copy()

    for frame, group
    in tracks.groupby("frame")
}


# ============================================================
# KEY FRAME CSV
# ============================================================

key_rows = []


for frame, description in KEY_FRAMES.items():

    if frame not in possession_by_frame.index:
        continue

    row = possession_by_frame.loc[
        frame
    ]

    if isinstance(
        row,
        pd.DataFrame
    ):
        row = row.iloc[0]

    key_rows.append(
        {
            "frame":
                frame,

            "time_sec":
                row.get(
                    "time_sec",
                    np.nan,
                ),

            "description":
                description,

            "possession_v5_3":
                row.get(
                    "possession_v5_3",
                    None,
                ),

            "ball_state_v5_3":
                row.get(
                    "ball_state_v5_3",
                    None,
                ),

            "controller_team_v5_3":
                row.get(
                    "controller_team_v5_3",
                    None,
                ),

            "controller_id_v5_3":
                row.get(
                    "controller_id_v5_3",
                    np.nan,
                ),

            "controller_gate_status":
                row.get(
                    "controller_gate_status",
                    None,
                ),

            "controller_gate_reason":
                row.get(
                    "controller_gate_reason",
                    None,
                ),

            "contest_episode_active":
                row.get(
                    "contest_episode_active",
                    False,
                ),

            "contest_episode_phase":
                row.get(
                    "contest_episode_phase",
                    None,
                ),

            "source_v5_3":
                row.get(
                    "source_v5_3",
                    None,
                ),
        }
    )


key_df = pd.DataFrame(
    key_rows
)


key_df.to_csv(
    OUTPUT_KEY_FRAMES,
    index=False,
)


# ============================================================
# PLAYER MARKER
# ============================================================

def find_player_point(
    frame,
    track_id,
):

    if (
        frame not in tracks_by_frame
        or
        pd.isna(track_id)
    ):
        return None


    group = tracks_by_frame[
        frame
    ]


    match = group[
        np.abs(
            group["track_id"]
            -
            float(track_id)
        )
        <
        0.1
    ]


    if len(match) == 0:
        return None


    row = match.iloc[0]


    return (
        int(row["image_x"]),
        int(row["image_y"]),
    )


# ============================================================
# OPEN VIDEO
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

    (width, height),
)


if not writer.isOpened():

    raise RuntimeError(
        f"Cannot create: {OUTPUT_VIDEO}"
    )


# ============================================================
# VIDEO LOOP
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V5.3 VISUAL QA"
)

print(
    "========================================"
)

print("")


frame_idx = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    annotated = frame.copy()


    # ========================================================
    # NO POSSESSION DATA
    # ========================================================

    if (
        frame_idx
        not in
        possession_by_frame.index
    ):

        writer.write(
            annotated
        )

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


    # ========================================================
    # VALUES
    # ========================================================

    team_possession = clean_text(
        row.get(
            "possession_v5_3",
            None,
        ),
        "Unknown",
    )


    ball_state = clean_text(
        row.get(
            "ball_state_v5_3",
            None,
        ),
        "Unknown",
    )


    controller_team = clean_team(
        row.get(
            "controller_team_v5_3",
            None,
        )
    )


    controller_id = clean_id(
        row.get(
            "controller_id_v5_3",
            np.nan,
        )
    )


    last_touch_team = clean_team(
        row.get(
            "last_touch_team_v5_3",
            None,
        )
    )


    last_touch_id = clean_id(
        row.get(
            "last_touch_id_v5_3",
            np.nan,
        )
    )


    last_touch_type = clean_text(
        row.get(
            "last_touch_type_v5_3",
            None,
        ),
        "NONE",
    )


    source = clean_text(
        row.get(
            "source_v5_3",
            None,
        ),
        "NONE",
    )


    gate_status = clean_text(
        row.get(
            "controller_gate_status",
            None,
        ),
        "NONE",
    )


    gate_reason = clean_text(
        row.get(
            "controller_gate_reason",
            None,
        ),
        "NONE",
    )


    gate_mode = clean_text(
        row.get(
            "controller_gate_mode",
            None,
        ),
        "NONE",
    )


    contest_active = as_bool(
        row.get(
            "contest_episode_active",
            False,
        )
    )


    contest_phase = clean_text(
        row.get(
            "contest_episode_phase",
            None,
        ),
        "NONE",
    )


    contest_participants = clean_text(
        row.get(
            "contest_episode_participants",
            None,
        ),
        "NONE",
    )


    ball_speed = row.get(
        "ball_speed_mps",
        np.nan,
    )


    ball_x = row.get(
        "ball_image_x",
        np.nan,
    )


    ball_y = row.get(
        "ball_image_y",
        np.nan,
    )


    # ========================================================
    # DRAW TRACKED PLAYERS
    # ========================================================

    if frame_idx in tracks_by_frame:

        frame_tracks = (
            tracks_by_frame[
                frame_idx
            ]
        )


        for _, player in (
            frame_tracks.iterrows()
        ):

            team = clean_team(
                player.get(
                    "stable_team",
                    None,
                )
            )


            if team is None:
                continue


            x = int(
                player[
                    "image_x"
                ]
            )

            y = int(
                player[
                    "image_y"
                ]
            )


            track_id = int(
                player[
                    "track_id"
                ]
            )


            if team == "A":

                player_color = (
                    COLOR_A
                )

            else:

                player_color = (
                    COLOR_B
                )


            cv2.circle(
                annotated,
                (x, y),
                PLAYER_RADIUS,
                player_color,
                -1,
            )


            put_text(
                annotated,
                f"{team}#{track_id}",
                x + 7,
                y - 7,
                scale=0.38,
                color=player_color,
                thickness=1,
            )


    # ========================================================
    # DRAW BALL
    # ========================================================

    if (
        pd.notna(ball_x)
        and
        pd.notna(ball_y)
    ):

        bx = int(ball_x)
        by = int(ball_y)


        cv2.circle(
            annotated,
            (bx, by),
            BALL_RADIUS,
            COLOR_BALL,
            3,
        )


        cv2.circle(
            annotated,
            (bx, by),
            3,
            COLOR_BALL,
            -1,
        )


        put_text(
            annotated,
            "BALL",
            bx + 13,
            by - 9,
            scale=0.42,
            color=COLOR_BALL,
        )


    # ========================================================
    # CONTROLLER
    # ========================================================

    controller_point = find_player_point(
        frame_idx,
        controller_id,
    )


    if (
        controller_point is not None
        and
        controller_team is not None
    ):

        cx, cy = (
            controller_point
        )


        cv2.circle(
            annotated,
            (cx, cy),
            CONTROLLER_RADIUS,
            COLOR_CONTROLLER,
            3,
        )


        put_text(
            annotated,
            "CTRL",
            cx + 14,
            cy + 17,
            scale=0.43,
            color=COLOR_CONTROLLER,
        )


    # ========================================================
    # LAST TOUCH
    # ========================================================

    last_touch_point = find_player_point(
        frame_idx,
        last_touch_id,
    )


    if (
        last_touch_point is not None
        and
        last_touch_team is not None
    ):

        lx, ly = (
            last_touch_point
        )


        cv2.circle(
            annotated,
            (lx, ly),
            LAST_TOUCH_RADIUS,
            COLOR_LAST_TOUCH,
            2,
        )


        put_text(
            annotated,
            "LT",
            lx - 20,
            ly + 20,
            scale=0.4,
            color=COLOR_LAST_TOUCH,
        )


    # ========================================================
    # PANEL
    # ========================================================

    draw_transparent_panel(
        annotated,

        PANEL_X,
        PANEL_Y,

        min(
            PANEL_X + PANEL_W,
            width - 15,
        ),

        PANEL_Y + PANEL_H,
    )


    y = 43


    put_text(
        annotated,
        (
            f"POSSESSION V5.3 QA | "
            f"frame={frame_idx} | "
            f"time="
            f"{row.get('time_sec', frame_idx / fps):.2f}s"
        ),
        30,
        y,
        scale=0.58,
    )


    y += 31


    # ========================================================
    # POSSESSION COLOR
    # ========================================================

    if team_possession == "A":

        possession_color = (
            COLOR_A
        )

    elif team_possession == "B":

        possession_color = (
            COLOR_B
        )

    elif team_possession == "Contested":

        possession_color = (
            COLOR_CONTEST
        )

    else:

        possession_color = (
            COLOR_UNKNOWN
        )


    put_text(
        annotated,
        (
            f"TEAM POSSESSION: "
            f"{team_possession}"
        ),
        30,
        y,
        scale=0.57,
        color=possession_color,
    )


    y += 29


    if ball_state == "AerialContest":

        state_color = (
            COLOR_CONTEST
        )

    elif ball_state == "Uncontrolled":

        state_color = (
            COLOR_UNCONTROLLED
        )

    elif ball_state == "Controlled":

        state_color = (
            COLOR_CONTROLLER
        )

    else:

        state_color = (
            COLOR_WHITE
        )


    put_text(
        annotated,
        (
            f"BALL STATE: "
            f"{ball_state}"
        ),
        30,
        y,
        scale=0.55,
        color=state_color,
    )


    y += 28


    put_text(
        annotated,
        (
            f"CONTROLLER: "
            f"{format_player(controller_team, controller_id)}"
        ),
        30,
        y,
        scale=0.52,
        color=(
            COLOR_CONTROLLER

            if controller_team
            is not None

            else COLOR_WHITE
        ),
    )


    y += 27


    put_text(
        annotated,
        (
            f"LAST TOUCH: "
            f"{format_player(last_touch_team, last_touch_id)} "
            f"({last_touch_type})"
        ),
        30,
        y,
        scale=0.48,
    )


    y += 27


    put_text(
        annotated,
        (
            f"SOURCE: "
            f"{source}"
        ),
        30,
        y,
        scale=0.44,
    )


    y += 26


    # ========================================================
    # GATE
    # ========================================================

    if gate_status != "NONE":

        if gate_status == "REJECT":

            gate_color = (
                COLOR_UNCONTROLLED
            )

        elif gate_status == "KEEP":

            gate_color = (
                COLOR_CONTROLLER
            )

        else:

            gate_color = (
                COLOR_CONTEST
            )


        put_text(
            annotated,
            (
                f"CONTROLLER GATE: "
                f"{gate_status} | "
                f"{gate_reason} | "
                f"{gate_mode}"
            ),
            30,
            y,
            scale=0.42,
            color=gate_color,
        )


    else:

        put_text(
            annotated,
            "CONTROLLER GATE: NONE",
            30,
            y,
            scale=0.42,
        )


    y += 26


    # ========================================================
    # CONTEST EPISODE
    # ========================================================

    if contest_active:

        put_text(
            annotated,
            (
                f"CONTEST EPISODE: "
                f"{contest_phase} | "
                f"{contest_participants}"
            ),
            30,
            y,
            scale=0.45,
            color=COLOR_CONTEST,
        )


    else:

        put_text(
            annotated,
            "CONTEST EPISODE: NONE",
            30,
            y,
            scale=0.42,
        )


    y += 26


    # ========================================================
    # SPEED
    # ========================================================

    if pd.notna(
        ball_speed
    ):

        put_text(
            annotated,
            (
                f"BALL SPEED: "
                f"{float(ball_speed):.2f} m/s"
            ),
            30,
            y,
            scale=0.44,
        )


    # ========================================================
    # V5.2 -> V5.3 CHANGE
    # ========================================================

    changed = as_bool(
        row.get(
            "contest_changed_from_v5_2",
            False,
        )
    )


    if changed:

        old_possession = clean_text(
            row.get(
                "possession_v5_2",
                None,
            ),
            "NONE",
        )


        old_state = clean_text(
            row.get(
                "ball_state_v5_2",
                None,
            ),
            "NONE",
        )


        cv2.rectangle(
            annotated,

            (15, 355),

            (
                min(
                    width - 15,
                    940,
                ),
                410,
            ),

            COLOR_CONTEST,

            -1,
        )


        put_text(
            annotated,
            (
                f"V5.2 -> V5.3 OVERRIDE | "
                f"{old_possession}/{old_state} "
                f"-> "
                f"{team_possession}/{ball_state}"
            ),
            30,
            389,
            scale=0.58,
            color=COLOR_BLACK,
        )


    # ========================================================
    # UNCONTROLLED BANNER
    # ========================================================

    elif ball_state == "Uncontrolled":

        cv2.rectangle(
            annotated,

            (15, 355),

            (
                min(
                    width - 15,
                    940,
                ),
                410,
            ),

            COLOR_UNCONTROLLED,

            -1,
        )


        put_text(
            annotated,
            (
                "CONTROLLER REMOVED | "
                "Control evidence rejected"
            ),
            30,
            389,
            scale=0.58,
            color=COLOR_WHITE,
        )


    # ========================================================
    # KEY FRAME BANNER
    # ========================================================

    if frame_idx in KEY_FRAMES:

        description = (
            KEY_FRAMES[
                frame_idx
            ]
        )


        cv2.rectangle(
            annotated,

            (
                15,
                height - 72,
            ),

            (
                min(
                    width - 15,
                    1000,
                ),
                height - 18,
            ),

            COLOR_BLACK,

            -1,
        )


        put_text(
            annotated,
            (
                f"QA KEY FRAME | "
                f"{description}"
            ),
            30,
            height - 38,
            scale=0.61,
            color=COLOR_BALL,
        )


    writer.write(
        annotated
    )


    frame_idx += 1


# ============================================================
# CLOSE
# ============================================================

cap.release()
writer.release()


# ============================================================
# PRINT QA WINDOWS
# ============================================================

print("")
print(
    "QA video:",
    OUTPUT_VIDEO
)

print(
    "Key-frame CSV:",
    OUTPUT_KEY_FRAMES
)


print("")
print(
    "========================================"
)

print(
    "PRIORITY VISUAL QA WINDOWS"
)

print(
    "========================================"
)

print("")

print(
    "1) 6.80 - 7.12 s"
)

print(
    "   Confirm 172-176 are truly one aerial contest"
)

print(
    "   Confirm frame 177 is a valid resolution"
)


print("")

print(
    "2) 8.48 - 8.56 s"
)

print(
    "   Inspect frame 213 hard rejection"
)


print("")

print(
    "3) 8.96 - 9.44 s"
)

print(
    "   Inspect frames 225-226 and 232-235"
)

print(
    "   All should genuinely have no controller"
)


print("")

print(
    "4) 13.36 - 13.60 s"
)

print(
    "   Inspect frames 335-339"
)

print(
    "   Verify upper-body geometry rejection"
)


print("")

print(
    "5) 25.84 - 26.08 s"
)

print(
    "   Verify A#11 dribble remains Controlled"
)
