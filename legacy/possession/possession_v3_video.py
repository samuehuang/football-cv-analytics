import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

PLAYER_CSV = "outputs/tracking_data_clean_v2.csv"
POSSESSION_CSV = "outputs/possession_v3_frames.csv"

OUTPUT_VIDEO = "outputs/possession_v3_qa.mp4"
OUTPUT_SWITCH_CSV = "outputs/possession_v3_turnovers.csv"
OUTPUT_MISMATCH_CSV = "outputs/possession_v3_projection_mismatch.csv"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Same stable-team rule as Possession V3
# ============================================================

MIN_DOMINANT_TEAM_SHARE = 0.70


# ============================================================
# Colors (BGR)
# ============================================================

TEAM_A_COLOR = (255, 120, 40)
TEAM_B_COLOR = (40, 80, 255)

BALL_COLOR = (0, 255, 255)

CONTESTED_COLOR = (0, 165, 255)
UNKNOWN_COLOR = (150, 150, 150)

IN_FLIGHT_COLOR = (255, 0, 255)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


# ============================================================
# Helpers
# ============================================================

def normalize_bool(series):

    return (
        series
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
    )


def state_color(state):

    if state == "A":
        return TEAM_A_COLOR

    if state == "B":
        return TEAM_B_COLOR

    if state == "Contested":
        return CONTESTED_COLOR

    return UNKNOWN_COLOR


def draw_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.68
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
# Load
# ============================================================

players = pd.read_csv(
    PLAYER_CSV
)

possession = pd.read_csv(
    POSSESSION_CSV
)


# ============================================================
# Normalize players
# ============================================================

for column in [
    "frame",
    "track_id",
    "image_x",
    "image_y"
]:

    players[column] = pd.to_numeric(
        players[column],
        errors="coerce"
    )


players = players.dropna(
    subset=[
        "frame",
        "track_id",
        "image_x",
        "image_y"
    ]
).copy()


players["frame"] = (
    players["frame"]
    .astype(int)
)


players = players[
    players["team"]
    .isin(
        [
            "A",
            "B"
        ]
    )
].copy()


# ============================================================
# Reconstruct stable team assignments
#
# Match the logic used by possession_v3.py
# ============================================================

track_team_stats = []


for track_id, group in players.groupby(
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

    dominant_count = int(
        counts.iloc[0]
    )

    total_count = len(group)

    dominant_share = (
        dominant_count
        /
        total_count
    )


    track_team_stats.append(
        {
            "track_id":
                track_id,

            "dominant_team":
                dominant_team,

            "dominant_share":
                dominant_share,
        }
    )


track_team_stats = pd.DataFrame(
    track_team_stats
)


stable_stats = track_team_stats[
    track_team_stats[
        "dominant_share"
    ]
    >=
    MIN_DOMINANT_TEAM_SHARE
].copy()


stable_team_map = dict(
    zip(
        stable_stats[
            "track_id"
        ],
        stable_stats[
            "dominant_team"
        ]
    )
)


players["stable_team"] = (
    players["track_id"]
    .map(
        stable_team_map
    )
)


players = players[
    players["stable_team"]
    .notna()
].copy()


# Remove inconsistent rows rather than relabeling them
players = players[
    players["team"]
    ==
    players["stable_team"]
].copy()


players["team"] = (
    players["stable_team"]
)


players_by_frame = {

    int(frame):
        group.copy()

    for frame, group
    in players.groupby("frame")
}


# ============================================================
# Normalize possession
# ============================================================

possession["frame"] = pd.to_numeric(
    possession["frame"],
    errors="coerce"
)


possession = possession.dropna(
    subset=["frame"]
).copy()


possession["frame"] = (
    possession["frame"]
    .astype(int)
)


for column in [
    "controller_id",

    "nearest_A_id",
    "nearest_B_id",

    "nearest_A_pitch_distance_m",
    "nearest_B_pitch_distance_m",

    "nearest_A_image_foot_distance_px",
    "nearest_B_image_foot_distance_px",

    "ball_image_x",
    "ball_image_y",
    "ball_pitch_x_m",
    "ball_pitch_y_m",
    "ball_speed_mps",

    "switch_candidate_count",
    "in_flight_streak",
]:

    if column in possession.columns:

        possession[column] = pd.to_numeric(
            possession[column],
            errors="coerce"
        )


possession["ball_usable_bool"] = (
    normalize_bool(
        possession["ball_usable"]
    )
)


possession["projection_mismatch_bool"] = (
    normalize_bool(
        possession[
            "projection_mismatch"
        ]
    )
)


possession["player_frame_good_bool"] = (
    normalize_bool(
        possession[
            "player_frame_good"
        ]
    )
)


possession_by_frame = (
    possession
    .set_index("frame")
)


MIN_FRAME = int(
    possession["frame"].min()
)

MAX_FRAME = int(
    possession["frame"].max()
)


# ============================================================
# Player lookup
# ============================================================

def find_player(
    frame_players,
    track_id
):

    if (
        frame_players is None
        or
        pd.isna(track_id)
    ):

        return None


    distance = np.abs(
        frame_players["track_id"]
        -
        float(track_id)
    )


    valid = distance.notna()


    if not valid.any():

        return None


    index = (
        distance[valid]
        .idxmin()
    )


    if (
        distance.loc[index]
        >
        0.1
    ):

        return None


    return frame_players.loc[
        index
    ]


# ============================================================
# Extract confirmed turnovers
# ============================================================

turnovers = []

previous_team = None


for _, row in possession.iterrows():

    state = str(
        row[
            "possession_v3"
        ]
    )

    source = str(
        row[
            "possession_source"
        ]
    )


    if source == "confirmed_turnover":

        turnovers.append(
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
                    previous_team,

                "to_team":
                    state,

                "direct_state":
                    row[
                        "direct_state"
                    ],

                "controller_id":
                    row[
                        "controller_id"
                    ],

                "nearest_A_pitch_m":
                    row[
                        "nearest_A_pitch_distance_m"
                    ],

                "nearest_B_pitch_m":
                    row[
                        "nearest_B_pitch_distance_m"
                    ],

                "nearest_A_foot_px":
                    row[
                        "nearest_A_image_foot_distance_px"
                    ],

                "nearest_B_foot_px":
                    row[
                        "nearest_B_image_foot_distance_px"
                    ],
            }
        )


    if state in [
        "A",
        "B"
    ]:

        previous_team = state


turnover_df = pd.DataFrame(
    turnovers
)


turnover_df.to_csv(
    OUTPUT_SWITCH_CSV,
    index=False
)


# ============================================================
# Projection mismatch diagnostic CSV
# ============================================================

mismatch_df = possession[
    possession[
        "projection_mismatch_bool"
    ]
].copy()


mismatch_columns = [

    "frame",
    "time_sec",

    "possession_v3",
    "direct_state",
    "possession_source",

    "nearest_A_id",
    "nearest_A_pitch_distance_m",
    "nearest_A_image_foot_distance_px",

    "nearest_B_id",
    "nearest_B_pitch_distance_m",
    "nearest_B_image_foot_distance_px",

    "ball_speed_mps",
]


mismatch_columns = [
    column
    for column in mismatch_columns
    if column in mismatch_df.columns
]


mismatch_df[
    mismatch_columns
].to_csv(
    OUTPUT_MISMATCH_CSV,
    index=False
)


# ============================================================
# Timeline
# ============================================================

timeline_states = {

    int(row["frame"]):
        str(
            row[
                "possession_v3"
            ]
        )

    for _, row
    in possession.iterrows()
}


def draw_timeline(
    frame,
    current_frame
):

    h, w = frame.shape[:2]


    margin = 30

    y1 = h - 42
    y2 = h - 19


    timeline_width = (
        w
        -
        margin * 2
    )


    cv2.rectangle(
        frame,
        (
            margin,
            y1
        ),
        (
            margin
            +
            timeline_width,
            y2
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


    for tactical_frame, state in (
        timeline_states.items()
    ):

        ratio = (
            tactical_frame
            -
            MIN_FRAME
        ) / frame_range


        x = int(
            margin
            +
            ratio
            *
            timeline_width
        )


        cv2.line(
            frame,
            (
                x,
                y1
            ),
            (
                x,
                y2
            ),
            state_color(
                state
            ),
            3
        )


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


        x = int(
            margin
            +
            ratio
            *
            timeline_width
        )


        cv2.line(
            frame,
            (
                x,
                y1 - 7
            ),
            (
                x,
                y2 + 7
            ),
            WHITE,
            2
        )


    cv2.putText(
        frame,
        "POSSESSION V3 TIMELINE",
        (
            margin,
            y1 - 7
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA
    )


# ============================================================
# Video
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
# Main loop
# ============================================================

frame_idx = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    # ========================================================
    # Outside V3 analysis range
    # ========================================================

    if (
        frame_idx
        not in
        possession_by_frame.index
    ):

        draw_panel(
            frame,
            15,
            15,
            500,
            88
        )


        cv2.putText(
            frame,
            "POSSESSION V3 QA",
            (
                30,
                47
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            WHITE,
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            "Outside analysis range",
            (
                30,
                75
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
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
    # Current possession row
    # ========================================================

    row = possession_by_frame.loc[
        frame_idx
    ]


    if isinstance(
        row,
        pd.DataFrame
    ):

        row = row.iloc[0]


    possession_state = str(
        row[
            "possession_v3"
        ]
    )


    direct_state = str(
        row[
            "direct_state"
        ]
    )


    possession_source = str(
        row[
            "possession_source"
        ]
    )


    state_colour = state_color(
        possession_state
    )


    frame_players = (
        players_by_frame.get(
            frame_idx,
            None
        )
    )


    # ========================================================
    # Draw all players
    # ========================================================

    if frame_players is not None:

        for _, player in (
            frame_players.iterrows()
        ):

            x = int(
                player["image_x"]
            )

            y = int(
                player["image_y"]
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


    if (
        row[
            "ball_usable_bool"
        ]
        and
        pd.notna(
            row[
                "ball_image_x"
            ]
        )
        and
        pd.notna(
            row[
                "ball_image_y"
            ]
        )
    ):

        ball_x = int(
            row[
                "ball_image_x"
            ]
        )

        ball_y = int(
            row[
                "ball_image_y"
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
                ball_y - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            BALL_COLOR,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Nearest A player
    # ========================================================

    nearest_A_id = row[
        "nearest_A_id"
    ]


    A_player = find_player(
        frame_players,
        nearest_A_id
    )


    if A_player is not None:

        ax = int(
            A_player[
                "image_x"
            ]
        )

        ay = int(
            A_player[
                "image_y"
            ]
        )


        cv2.circle(
            frame,
            (
                ax,
                ay
            ),
            12,
            TEAM_A_COLOR,
            3
        )


        cv2.putText(
            frame,
            (
                f"A #{int(nearest_A_id)}"
            ),
            (
                ax + 14,
                ay - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            TEAM_A_COLOR,
            2,
            cv2.LINE_AA
        )


        if ball_x is not None:

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
    # Nearest B player
    # ========================================================

    nearest_B_id = row[
        "nearest_B_id"
    ]


    B_player = find_player(
        frame_players,
        nearest_B_id
    )


    if B_player is not None:

        bx = int(
            B_player[
                "image_x"
            ]
        )

        by = int(
            B_player[
                "image_y"
            ]
        )


        cv2.circle(
            frame,
            (
                bx,
                by
            ),
            12,
            TEAM_B_COLOR,
            3
        )


        cv2.putText(
            frame,
            (
                f"B #{int(nearest_B_id)}"
            ),
            (
                bx + 14,
                by - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            TEAM_B_COLOR,
            2,
            cv2.LINE_AA
        )


        if ball_x is not None:

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
    # Controller
    # ========================================================

    controller_id = row[
        "controller_id"
    ]


    controller = find_player(
        frame_players,
        controller_id
    )


    if controller is not None:

        cx = int(
            controller[
                "image_x"
            ]
        )

        cy = int(
            controller[
                "image_y"
            ]
        )


        cv2.circle(
            frame,
            (
                cx,
                cy
            ),
            18,
            GREEN,
            3
        )


        cv2.putText(
            frame,
            "CONTROLLER",
            (
                cx + 20,
                cy + 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            GREEN,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Information panel
    # ========================================================

    draw_panel(
        frame,
        15,
        15,
        790,
        285
    )


    cv2.putText(
        frame,
        (
            f"POSSESSION V3: "
            f"{possession_state}"
        ),
        (
            30,
            49
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        state_colour,
        2,
        cv2.LINE_AA
    )


    direct_color = (
        IN_FLIGHT_COLOR
        if direct_state == "InFlight"
        else WHITE
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
        direct_color,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        (
            f"Source: "
            f"{possession_source}"
        ),
        (
            30,
            108
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        WHITE,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Controller text
    # ========================================================

    if (
        pd.notna(
            controller_id
        )
        and
        str(
            row[
                "controller_team"
            ]
        )
        in [
            "A",
            "B"
        ]
    ):

        controller_text = (
            f"Controller: "
            f"{row['controller_team']} "
            f"#{int(controller_id)}"
        )

    else:

        controller_text = (
            "Controller: NONE"
        )


    cv2.putText(
        frame,
        controller_text,
        (
            30,
            136
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        GREEN,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # Nearest-player metrics
    # ========================================================

    A_pitch = row[
        "nearest_A_pitch_distance_m"
    ]

    A_foot = row[
        "nearest_A_image_foot_distance_px"
    ]


    B_pitch = row[
        "nearest_B_pitch_distance_m"
    ]

    B_foot = row[
        "nearest_B_image_foot_distance_px"
    ]


    if (
        pd.notna(A_pitch)
        and
        pd.notna(A_foot)
    ):

        A_text = (
            f"Nearest A #{int(nearest_A_id)} | "
            f"pitch={A_pitch:.2f}m | "
            f"foot={A_foot:.0f}px"
        )

    else:

        A_text = (
            "Nearest A: unavailable"
        )


    if (
        pd.notna(B_pitch)
        and
        pd.notna(B_foot)
    ):

        B_text = (
            f"Nearest B #{int(nearest_B_id)} | "
            f"pitch={B_pitch:.2f}m | "
            f"foot={B_foot:.0f}px"
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
            166
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEAM_A_COLOR,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        B_text,
        (
            30,
            195
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEAM_B_COLOR,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # Ball info
    # ========================================================

    ball_quality = str(
        row[
            "ball_quality"
        ]
    )


    ball_speed = row[
        "ball_speed_mps"
    ]


    if pd.notna(
        ball_speed
    ):

        ball_info = (
            f"Ball: {ball_quality} | "
            f"speed={ball_speed:.2f} m/s"
        )

    else:

        ball_info = (
            f"Ball: {ball_quality}"
        )


    cv2.putText(
        frame,
        ball_info,
        (
            30,
            224
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        BALL_COLOR,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Player counts
    # ========================================================

    cv2.putText(
        frame,
        (
            f"Players: "
            f"A={int(row['A_player_count'])} "
            f"B={int(row['B_player_count'])}"
        ),
        (
            30,
            253
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        WHITE,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # Projection mismatch banner
    # ========================================================

    if row[
        "projection_mismatch_bool"
    ]:

        text = (
            "AERIAL / PROJECTION MISMATCH"
        )


        cv2.rectangle(
            frame,
            (
                width - 570,
                20
            ),
            (
                width - 20,
                72
            ),
            IN_FLIGHT_COLOR,
            -1
        )


        cv2.putText(
            frame,
            text,
            (
                width - 550,
                55
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            BLACK,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # In-flight banner
    # ========================================================

    elif direct_state == "InFlight":

        cv2.putText(
            frame,
            (
                f"IN FLIGHT | "
                f"retain Team "
                f"{possession_state}"
            ),
            (
                width - 470,
                45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            IN_FLIGHT_COLOR,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Pending turnover
    # ========================================================

    switch_candidate = row.get(
        "switch_candidate",
        None
    )


    switch_count = row.get(
        "switch_candidate_count",
        0
    )


    if (
        pd.notna(
            switch_candidate
        )
        and
        str(
            switch_candidate
        )
        not in [
            "None",
            "nan"
        ]
    ):

        cv2.putText(
            frame,
            (
                f"TURNOVER CANDIDATE: "
                f"{switch_candidate} "
                f"{int(switch_count)}/4"
            ),
            (
                width - 520,
                92
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
        possession_source
        ==
        "confirmed_turnover"
    ):

        banner = (
            f"CONFIRMED TURNOVER "
            f"-> TEAM "
            f"{possession_state}"
        )


        text_size = cv2.getTextSize(
            banner,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.84,
            2
        )[0]


        x = (
            width // 2
            -
            text_size[0] // 2
        )


        cv2.rectangle(
            frame,
            (
                x - 25,
                310
            ),
            (
                x
                +
                text_size[0]
                +
                25,
                365
            ),
            state_colour,
            -1
        )


        cv2.putText(
            frame,
            banner,
            (
                x,
                347
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.84,
            BLACK,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Frame/time
    # ========================================================

    cv2.putText(
        frame,
        (
            f"Frame {frame_idx} | "
            f"{frame_idx / fps:.2f}s"
        ),
        (
            width - 310,
            height - 61
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
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
# Console summary
# ============================================================

print("")
print(
    "========================================"
)

print(
    "POSSESSION V3 VISUAL QA"
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
    "Turnover CSV:",
    OUTPUT_SWITCH_CSV
)

print(
    "Projection mismatch CSV:",
    OUTPUT_MISMATCH_CSV
)

print("")

print(
    "Window:",
    MIN_FRAME,
    "->",
    MAX_FRAME,
    f"({MIN_FRAME / fps:.2f}s -> "
    f"{MAX_FRAME / fps:.2f}s)"
)

print("")

print(
    "Confirmed turnovers:",
    len(turnover_df)
)


if len(turnover_df) > 0:

    print("")

    print(
        turnover_df
        .round(2)
        .to_string(
            index=False
        )
    )


print("")

print(
    "Projection mismatch frames:",
    len(mismatch_df)
)


if len(mismatch_df) > 0:

    print("")

    print(
        mismatch_df[
            mismatch_columns
        ]
        .head(20)
        .round(2)
        .to_string(
            index=False
        )
    )


print("")
