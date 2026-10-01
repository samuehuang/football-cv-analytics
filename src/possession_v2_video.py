import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

PLAYER_CSV = "outputs/tracking_data_tactical.csv"
BALL_CSV = "outputs/ball_tracking_ensemble_trusted.csv"
POSSESSION_CSV = "outputs/possession_v2_frames.csv"

OUTPUT_VIDEO = "outputs/possession_v2_qa.mp4"
OUTPUT_SWITCH_CSV = "outputs/possession_v2_switches.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Colors (BGR)
# ============================================================

TEAM_A_COLOR = (255, 120, 40)
TEAM_B_COLOR = (40, 60, 255)

BALL_COLOR = (0, 255, 255)

CONTESTED_COLOR = (0, 165, 255)
UNKNOWN_COLOR = (140, 140, 140)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GREEN = (0, 255, 0)


# ============================================================
# Load data
# ============================================================

players = pd.read_csv(
    PLAYER_CSV
)

ball = pd.read_csv(
    BALL_CSV
)

possession = pd.read_csv(
    POSSESSION_CSV
)


# ============================================================
# Normalize frame columns
# ============================================================

players["frame"] = pd.to_numeric(
    players["frame"],
    errors="coerce"
)

ball["frame"] = pd.to_numeric(
    ball["frame"],
    errors="coerce"
)

possession["frame"] = pd.to_numeric(
    possession["frame"],
    errors="coerce"
)


players = players.dropna(
    subset=["frame"]
)

ball = ball.dropna(
    subset=["frame"]
)

possession = possession.dropna(
    subset=["frame"]
)


players["frame"] = (
    players["frame"]
    .astype(int)
)

ball["frame"] = (
    ball["frame"]
    .astype(int)
)

possession["frame"] = (
    possession["frame"]
    .astype(int)
)


# ============================================================
# Track ID normalization
# ============================================================

players["track_id"] = pd.to_numeric(
    players["track_id"],
    errors="coerce"
)


for column in [
    "nearest_A_id",
    "nearest_B_id",
    "nearest_player_id",
]:

    if column in possession.columns:

        possession[column] = pd.to_numeric(
            possession[column],
            errors="coerce"
        )


# ============================================================
# Find player image coordinate columns
# ============================================================

IMAGE_X_OPTIONS = [
    "image_x",
    "image_x_clean",
    "center_x",
]

IMAGE_Y_OPTIONS = [
    "image_y",
    "image_y_clean",
    "center_y",
]


PLAYER_IMAGE_X = next(
    (
        c
        for c in IMAGE_X_OPTIONS
        if c in players.columns
    ),
    None
)

PLAYER_IMAGE_Y = next(
    (
        c
        for c in IMAGE_Y_OPTIONS
        if c in players.columns
    ),
    None
)


if (
    PLAYER_IMAGE_X is None
    or
    PLAYER_IMAGE_Y is None
):

    raise RuntimeError(
        "Cannot find player image coordinates.\n"
        f"Available columns:\n"
        f"{players.columns.tolist()}"
    )


print(
    "Player image coordinates:",
    PLAYER_IMAGE_X,
    PLAYER_IMAGE_Y
)


# ============================================================
# Normalize ball usable
# ============================================================

if "ball_usable" in ball.columns:

    ball["ball_usable_bool"] = (
        ball["ball_usable"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1"
            ]
        )
    )

else:

    ball["ball_usable_bool"] = False


# ============================================================
# Index data by frame
# ============================================================

ball_by_frame = (
    ball
    .set_index("frame")
)

possession_by_frame = (
    possession
    .set_index("frame")
)


players_by_frame = {
    int(frame):
        group.copy()

    for frame, group
    in players.groupby("frame")
}


# ============================================================
# Tactical window
# ============================================================

MIN_FRAME = int(
    possession["frame"].min()
)

MAX_FRAME = int(
    possession["frame"].max()
)


# ============================================================
# State color
# ============================================================

def get_state_color(
    state
):

    if state == "A":

        return TEAM_A_COLOR

    if state == "B":

        return TEAM_B_COLOR

    if state == "Contested":

        return CONTESTED_COLOR

    return UNKNOWN_COLOR


# ============================================================
# Player lookup
# ============================================================

def find_player(
    frame_players,
    team,
    track_id
):

    if (
        frame_players is None
        or
        pd.isna(track_id)
    ):

        return None


    candidates = frame_players[
        frame_players["team"]
        ==
        team
    ]


    if len(candidates) == 0:

        return None


    target_id = float(
        track_id
    )


    diff = np.abs(
        candidates["track_id"]
        -
        target_id
    )


    valid = diff.notna()


    if not valid.any():

        return None


    nearest_index = (
        diff[valid]
        .idxmin()
    )


    if (
        diff.loc[
            nearest_index
        ]
        >
        0.1
    ):

        return None


    return candidates.loc[
        nearest_index
    ]


# ============================================================
# Transparent panel
# ============================================================

def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.62
):

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        BLACK,
        -1
    )


    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
        frame
    )


# ============================================================
# Timeline
# ============================================================

timeline_states = {}


for _, row in possession.iterrows():

    timeline_states[
        int(row["frame"])
    ] = str(
        row["possession_v2"]
    )


def draw_timeline(
    frame,
    current_frame
):

    h, w = frame.shape[:2]


    margin_x = 35

    timeline_y1 = (
        h - 42
    )

    timeline_y2 = (
        h - 20
    )


    timeline_width = (
        w
        -
        2 * margin_x
    )


    # background
    cv2.rectangle(
        frame,
        (
            margin_x,
            timeline_y1
        ),
        (
            margin_x
            +
            timeline_width,
            timeline_y2
        ),
        (
            50,
            50,
            50
        ),
        -1
    )


    frame_range = max(
        MAX_FRAME
        -
        MIN_FRAME,
        1
    )


    # --------------------------------------------------------
    # Draw possession states
    # --------------------------------------------------------

    for tactical_frame, state in (
        timeline_states.items()
    ):

        ratio = (
            tactical_frame
            -
            MIN_FRAME
        ) / frame_range


        x = int(
            margin_x
            +
            ratio
            *
            timeline_width
        )


        color = get_state_color(
            state
        )


        cv2.line(
            frame,
            (
                x,
                timeline_y1
            ),
            (
                x,
                timeline_y2
            ),
            color,
            3
        )


    # --------------------------------------------------------
    # Current frame marker
    # --------------------------------------------------------

    if (
        MIN_FRAME
        <=
        current_frame
        <=
        MAX_FRAME
    ):

        ratio = (
            current_frame
            -
            MIN_FRAME
        ) / frame_range


        current_x = int(
            margin_x
            +
            ratio
            *
            timeline_width
        )


        cv2.line(
            frame,
            (
                current_x,
                timeline_y1 - 6
            ),
            (
                current_x,
                timeline_y2 + 6
            ),
            WHITE,
            2
        )


    cv2.putText(
        frame,
        "A",
        (
            margin_x,
            timeline_y1 - 7
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEAM_A_COLOR,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "B",
        (
            margin_x + 28,
            timeline_y1 - 7
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEAM_B_COLOR,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "POSSESSION TIMELINE",
        (
            margin_x + 55,
            timeline_y1 - 7
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA
    )


# ============================================================
# Extract confirmed switches
# ============================================================

switch_rows = []

previous_known_team = None


for _, row in possession.iterrows():

    current_state = str(
        row["possession_v2"]
    )

    source = str(
        row["possession_source"]
    )


    if (
        source
        ==
        "confirmed_switch"
    ):

        switch_rows.append(
            {
                "frame":
                    int(
                        row["frame"]
                    ),

                "time_sec":
                    float(
                        row["time_sec"]
                    ),

                "from_team":
                    previous_known_team,

                "to_team":
                    current_state,

                "direct_state":
                    row[
                        "direct_state"
                    ],

                "nearest_A_distance_m":
                    row[
                        "nearest_A_distance_m"
                    ],

                "nearest_B_distance_m":
                    row[
                        "nearest_B_distance_m"
                    ],
            }
        )


    if current_state in [
        "A",
        "B"
    ]:

        previous_known_team = (
            current_state
        )


switch_df = pd.DataFrame(
    switch_rows
)


switch_df.to_csv(
    OUTPUT_SWITCH_CSV,
    index=False
)


# ============================================================
# Video setup
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: "
        f"{VIDEO_PATH}"
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
        f"Cannot create video: "
        f"{OUTPUT_VIDEO}"
    )


# ============================================================
# Main video loop
# ============================================================

frame_idx = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    # ========================================================
    # Outside tactical QA frame
    # ========================================================

    if frame_idx not in possession_by_frame.index:

        draw_panel(
            frame,
            15,
            15,
            500,
            92
        )


        cv2.putText(
            frame,
            "POSSESSION V2 QA",
            (
                30,
                45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            WHITE,
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "Outside high-confidence tactical frame",
            (
                30,
                75
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            UNKNOWN_COLOR,
            2,
            cv2.LINE_AA
        )


        draw_timeline(
            frame,
            frame_idx
        )


        writer.write(
            frame
        )

        frame_idx += 1

        continue


    # ========================================================
    # Possession row
    # ========================================================

    possession_row = (
        possession_by_frame
        .loc[frame_idx]
    )


    if isinstance(
        possession_row,
        pd.DataFrame
    ):

        possession_row = (
            possession_row
            .iloc[0]
        )


    final_state = str(
        possession_row[
            "possession_v2"
        ]
    )


    direct_state = str(
        possession_row[
            "direct_state"
        ]
    )


    source = str(
        possession_row[
            "possession_source"
        ]
    )


    state_color = get_state_color(
        final_state
    )


    # ========================================================
    # Current players
    # ========================================================

    frame_players = (
        players_by_frame.get(
            frame_idx,
            None
        )
    )


    # --------------------------------------------------------
    # Draw all tactical players
    # --------------------------------------------------------

    if frame_players is not None:

        for _, player in (
            frame_players.iterrows()
        ):

            if (
                pd.isna(
                    player[
                        PLAYER_IMAGE_X
                    ]
                )
                or
                pd.isna(
                    player[
                        PLAYER_IMAGE_Y
                    ]
                )
            ):

                continue


            x = int(
                player[
                    PLAYER_IMAGE_X
                ]
            )

            y = int(
                player[
                    PLAYER_IMAGE_Y
                ]
            )


            team = str(
                player["team"]
            )


            color = (
                TEAM_A_COLOR
                if team == "A"
                else TEAM_B_COLOR
            )


            cv2.circle(
                frame,
                (
                    x,
                    y
                ),
                4,
                color,
                -1
            )


    # ========================================================
    # Trusted ball
    # ========================================================

    ball_x = None
    ball_y = None
    ball_quality = "missing"


    if frame_idx in ball_by_frame.index:

        ball_row = (
            ball_by_frame
            .loc[frame_idx]
        )


        if isinstance(
            ball_row,
            pd.DataFrame
        ):

            ball_row = (
                ball_row.iloc[0]
            )


        usable = bool(
            ball_row[
                "ball_usable_bool"
            ]
        )


        if (
            usable
            and
            pd.notna(
                ball_row[
                    "image_x_trusted"
                ]
            )
            and
            pd.notna(
                ball_row[
                    "image_y_trusted"
                ]
            )
        ):

            ball_x = int(
                ball_row[
                    "image_x_trusted"
                ]
            )

            ball_y = int(
                ball_row[
                    "image_y_trusted"
                ]
            )


            ball_quality = str(
                ball_row[
                    "ball_quality"
                ]
            )


            cv2.circle(
                frame,
                (
                    ball_x,
                    ball_y
                ),
                9,
                BALL_COLOR,
                -1
            )


            cv2.circle(
                frame,
                (
                    ball_x,
                    ball_y
                ),
                15,
                BALL_COLOR,
                2
            )


            cv2.putText(
                frame,
                "BALL",
                (
                    ball_x + 13,
                    ball_y - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                BALL_COLOR,
                2,
                cv2.LINE_AA
            )


    # ========================================================
    # Nearest Team A player
    # ========================================================

    nearest_A_id = possession_row[
        "nearest_A_id"
    ]


    nearest_A_distance = possession_row[
        "nearest_A_distance_m"
    ]


    nearest_A_player = find_player(
        frame_players,
        "A",
        nearest_A_id
    )


    if nearest_A_player is not None:

        ax = int(
            nearest_A_player[
                PLAYER_IMAGE_X
            ]
        )

        ay = int(
            nearest_A_player[
                PLAYER_IMAGE_Y
            ]
        )


        cv2.circle(
            frame,
            (
                ax,
                ay
            ),
            11,
            TEAM_A_COLOR,
            3
        )


        cv2.putText(
            frame,
            (
                f"A #{int(nearest_A_id)}"
            ),
            (
                ax + 13,
                ay - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            TEAM_A_COLOR,
            2,
            cv2.LINE_AA
        )


        if (
            ball_x is not None
            and
            pd.notna(
                nearest_A_distance
            )
        ):

            cv2.line(
                frame,
                (
                    ax,
                    ay
                ),
                (
                    ball_x,
                    ball_y
                ),
                TEAM_A_COLOR,
                1
            )


    # ========================================================
    # Nearest Team B player
    # ========================================================

    nearest_B_id = possession_row[
        "nearest_B_id"
    ]


    nearest_B_distance = possession_row[
        "nearest_B_distance_m"
    ]


    nearest_B_player = find_player(
        frame_players,
        "B",
        nearest_B_id
    )


    if nearest_B_player is not None:

        bx = int(
            nearest_B_player[
                PLAYER_IMAGE_X
            ]
        )

        by = int(
            nearest_B_player[
                PLAYER_IMAGE_Y
            ]
        )


        cv2.circle(
            frame,
            (
                bx,
                by
            ),
            11,
            TEAM_B_COLOR,
            3
        )


        cv2.putText(
            frame,
            (
                f"B #{int(nearest_B_id)}"
            ),
            (
                bx + 13,
                by - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            TEAM_B_COLOR,
            2,
            cv2.LINE_AA
        )


        if (
            ball_x is not None
            and
            pd.notna(
                nearest_B_distance
            )
        ):

            cv2.line(
                frame,
                (
                    bx,
                    by
                ),
                (
                    ball_x,
                    ball_y
                ),
                TEAM_B_COLOR,
                1
            )


    # ========================================================
    # Main possession panel
    # ========================================================

    draw_panel(
        frame,
        15,
        15,
        680,
        210
    )


    cv2.putText(
        frame,
        (
            f"POSSESSION V2: "
            f"{final_state}"
        ),
        (
            30,
            48
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        state_color,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        (
            f"Direct state: "
            f"{direct_state}"
        ),
        (
            30,
            80
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        (
            f"Source: "
            f"{source}"
        ),
        (
            30,
            108
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        WHITE,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Distance labels
    # ========================================================

    if pd.notna(
        nearest_A_distance
    ):

        A_text = (
            f"Nearest A: "
            f"#{int(nearest_A_id)}  "
            f"{nearest_A_distance:.2f} m"
        )

    else:

        A_text = (
            "Nearest A: unavailable"
        )


    if pd.notna(
        nearest_B_distance
    ):

        B_text = (
            f"Nearest B: "
            f"#{int(nearest_B_id)}  "
            f"{nearest_B_distance:.2f} m"
        )

    else:

        B_text = (
            "Nearest B: unavailable"
        )


    cv2.putText(
        frame,
        A_text,
        (
            30,
            139
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        TEAM_A_COLOR,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        B_text,
        (
            30,
            168
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        TEAM_B_COLOR,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        (
            f"Ball quality: "
            f"{ball_quality}"
        ),
        (
            30,
            197
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        BALL_COLOR,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Switch candidate
    # ========================================================

    switch_candidate = possession_row.get(
        "switch_candidate",
        None
    )


    switch_count = possession_row.get(
        "switch_candidate_count",
        0
    )


    if (
        pd.notna(switch_candidate)
        and
        str(switch_candidate)
        not in [
            "None",
            "nan"
        ]
    ):

        cv2.putText(
            frame,
            (
                f"Switch candidate: "
                f"{switch_candidate} "
                f"{int(switch_count)}/4"
            ),
            (
                width - 430,
                45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            CONTESTED_COLOR,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Confirmed turnover banner
    # ========================================================

    if (
        source
        ==
        "confirmed_switch"
    ):

        banner_text = (
            f"CONFIRMED TURNOVER -> TEAM "
            f"{final_state}"
        )


        text_size = cv2.getTextSize(
            banner_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            2
        )[0]


        banner_x = (
            width // 2
            -
            text_size[0] // 2
        )


        cv2.rectangle(
            frame,
            (
                banner_x - 20,
                235
            ),
            (
                banner_x
                +
                text_size[0]
                +
                20,
                285
            ),
            state_color,
            -1
        )


        cv2.putText(
            frame,
            banner_text,
            (
                banner_x,
                269
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            BLACK,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Time / frame
    # ========================================================

    cv2.putText(
        frame,
        (
            f"Frame {frame_idx} | "
            f"{frame_idx / fps:.2f}s"
        ),
        (
            width - 300,
            height - 62
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Timeline
    # ========================================================

    draw_timeline(
        frame,
        frame_idx
    )


    writer.write(
        frame
    )


    frame_idx += 1


# ============================================================
# Finish
# ============================================================

cap.release()
writer.release()


# ============================================================
# Print switches
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V2 VISUAL QA"
)

print(
    "========================================"
)

print("")

print(
    "QA Video:",
    OUTPUT_VIDEO
)

print(
    "Switch CSV:",
    OUTPUT_SWITCH_CSV
)

print("")

print(
    "Tactical window:"
)

print(
    f"{MIN_FRAME} -> {MAX_FRAME}"
)

print(
    f"{MIN_FRAME / fps:.2f}s "
    f"-> "
    f"{MAX_FRAME / fps:.2f}s"
)

print("")

print(
    "Confirmed switches:",
    len(switch_df)
)


if len(switch_df) > 0:

    print("")

    print(
        switch_df.round(3).to_string(
            index=False
        )
    )

print("")
