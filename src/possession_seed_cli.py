#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Validated V3 Pass-1 parameters
# ============================================================

MIN_PLAYERS_PER_TEAM = 6
MIN_DOMINANT_TEAM_SHARE = 0.70

CONTROL_RADIUS_M = 4.0
FOOT_CONTROL_MAX_PX = 95.0

CONTESTED_DISTANCE_MARGIN_M = 0.75
CONTESTED_FOOT_MARGIN_PX = 35.0


# ============================================================
# Validated V4 parameters
# ============================================================

RECEPTION_MAX_SPEED_MPS = 10.0
RECEPTION_CONFIRM_FRAMES = 3

IN_TRANSIT_SPEED_MPS = 10.0
MAX_IN_TRANSIT_HOLD_FRAMES = 50

AERIAL_CONTEST_MIN_SPEED_MPS = 8.0

AERIAL_CONTEST_PITCH_RADIUS_M = 6.0
AERIAL_CONTEST_IMAGE_RADIUS_PX = 180.0

AERIAL_CONTEST_PITCH_MARGIN_M = 1.5
AERIAL_CONTEST_IMAGE_MARGIN_PX = 70.0

LOOSE_HOLD_FRAMES = 12
BALL_MISSING_HOLD_FRAMES = 8
PLAYER_QUALITY_HOLD_FRAMES = 5


# ============================================================
# General helpers
# ============================================================

def require_file(path, label):

    path = Path(path)

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: "
            f"{path}"
        )

    if path.stat().st_size == 0:

        raise RuntimeError(
            f"{label} is empty: "
            f"{path}"
        )


def ensure_parent(path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def normalize_bool(series):

    return (
        series
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


# ============================================================
# Detect player pitch coordinate columns
# ============================================================

def detect_player_pitch_columns(
    players,
):

    x_options = [
        "pitch_x_final_m",
        "pitch_x_clean_m",
        "pitch_x_m",
    ]

    y_options = [
        "pitch_y_final_m",
        "pitch_y_clean_m",
        "pitch_y_m",
    ]


    player_x = next(
        (
            column
            for column
            in x_options
            if column
            in players.columns
        ),
        None,
    )


    player_y = next(
        (
            column
            for column
            in y_options
            if column
            in players.columns
        ),
        None,
    )


    if (
        player_x is None
        or
        player_y is None
    ):

        raise RuntimeError(
            "Cannot find player pitch "
            "coordinate columns.\n"
            f"Available columns:\n"
            f"{players.columns.tolist()}"
        )


    if (
        "image_x"
        not in
        players.columns

        or

        "image_y"
        not in
        players.columns
    ):

        raise RuntimeError(
            "Player tracking CSV must "
            "contain image_x and image_y."
        )


    return (
        player_x,
        player_y,
    )


# ============================================================
# V3 nearest-player helper
# ============================================================

def nearest_player(
    frame_players,
    team,
    ball_pitch_x,
    ball_pitch_y,
    ball_image_x,
    ball_image_y,
    player_x,
    player_y,
):

    team_players = (
        frame_players[
            frame_players[
                "team"
            ]
            ==
            team
        ]
        .copy()
    )


    team_players = (
        team_players
        .dropna(
            subset=[
                player_x,
                player_y,
                "image_x",
                "image_y",
            ]
        )
    )


    if len(team_players) == 0:

        return {
            "track_id":
                np.nan,

            "pitch_distance_m":
                np.nan,

            "image_foot_distance_px":
                np.nan,

            "direct_control":
                False,
        }


    pitch_dx = (
        team_players[
            player_x
        ]
        -
        ball_pitch_x
    )


    pitch_dy = (
        team_players[
            player_y
        ]
        -
        ball_pitch_y
    )


    pitch_distance = np.sqrt(
        pitch_dx ** 2
        +
        pitch_dy ** 2
    )


    image_dx = (
        team_players[
            "image_x"
        ]
        -
        ball_image_x
    )


    image_dy = (
        team_players[
            "image_y"
        ]
        -
        ball_image_y
    )


    image_distance = np.sqrt(
        image_dx ** 2
        +
        image_dy ** 2
    )


    valid = (
        pitch_distance.notna()
        &
        image_distance.notna()
    )


    if not valid.any():

        return {
            "track_id":
                np.nan,

            "pitch_distance_m":
                np.nan,

            "image_foot_distance_px":
                np.nan,

            "direct_control":
                False,
        }


    nearest_index = (
        pitch_distance[
            valid
        ]
        .idxmin()
    )


    pitch_dist = float(
        pitch_distance.loc[
            nearest_index
        ]
    )


    image_dist = float(
        image_distance.loc[
            nearest_index
        ]
    )


    player_row = (
        team_players.loc[
            nearest_index
        ]
    )


    direct_control = (

        pitch_dist
        <=
        CONTROL_RADIUS_M

        and

        image_dist
        <=
        FOOT_CONTROL_MAX_PX
    )


    return {
        "track_id":
            player_row[
                "track_id"
            ],

        "pitch_distance_m":
            pitch_dist,

        "image_foot_distance_px":
            image_dist,

        "direct_control":
            direct_control,
    }


# ============================================================
# PASS A
#
# Possession V3 PASS 1 only.
#
# Player + ball
#   ->
# geometric / nearest-player evidence
#
# V3 PASS 2 possession state machine is intentionally
# NOT included.
# ============================================================

def build_geometric_seed(
    players,
    ball,
    fps,
):

    (
        player_x,
        player_y,

    ) = detect_player_pitch_columns(
        players
    )


    print(
        "Using player pitch coordinates:",
        player_x,
        player_y,
    )


    # ========================================================
    # Normalize players
    # ========================================================

    for column in [
        "frame",
        "track_id",
        player_x,
        player_y,
        "image_x",
        "image_y",
    ]:

        players[
            column
        ] = pd.to_numeric(
            players[
                column
            ],
            errors="coerce",
        )


    players = (
        players
        .dropna(
            subset=[
                "frame",
                "track_id",
            ]
        )
        .copy()
    )


    players[
        "frame"
    ] = (
        players[
            "frame"
        ]
        .astype(int)
    )


    players = (
        players[
            players[
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
    # Stable team assignment
    # ========================================================

    track_team_stats = []


    for (
        track_id,
        group,

    ) in players.groupby(
        "track_id"
    ):

        team_counts = (
            group[
                "team"
            ]
            .value_counts()
        )


        if len(team_counts) == 0:

            continue


        dominant_team = (
            team_counts.index[
                0
            ]
        )


        dominant_count = int(
            team_counts.iloc[
                0
            ]
        )


        total_count = int(
            len(group)
        )


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

                "rows":
                    total_count,
            }
        )


    track_team_stats = (
        pd.DataFrame(
            track_team_stats
        )
    )


    if len(track_team_stats) == 0:

        raise RuntimeError(
            "No stable A/B player "
            "tracks were found."
        )


    stable_track_stats = (
        track_team_stats[
            track_team_stats[
                "dominant_share"
            ]
            >=
            MIN_DOMINANT_TEAM_SHARE
        ]
        .copy()
    )


    stable_track_ids = set(
        stable_track_stats[
            "track_id"
        ]
        .tolist()
    )


    dominant_team_map = dict(
        zip(
            stable_track_stats[
                "track_id"
            ],
            stable_track_stats[
                "dominant_team"
            ],
        )
    )


    players = (
        players[
            players[
                "track_id"
            ].isin(
                stable_track_ids
            )
        ]
        .copy()
    )


    players[
        "stable_team"
    ] = (
        players[
            "track_id"
        ]
        .map(
            dominant_team_map
        )
    )


    team_consistent_mask = (

        players[
            "team"
        ]

        ==

        players[
            "stable_team"
        ]
    )


    removed_inconsistent_rows = int(
        (
            ~team_consistent_mask
        ).sum()
    )


    players = (
        players[
            team_consistent_mask
        ]
        .copy()
    )


    players[
        "team"
    ] = (
        players[
            "stable_team"
        ]
    )


    # ========================================================
    # Normalize ball
    # ========================================================

    required_ball_columns = [
        "frame",
        "ball_usable",
        "pitch_x_trusted_m",
        "pitch_y_trusted_m",
        "image_x_trusted",
        "image_y_trusted",
        "trusted_speed_mps",
    ]


    missing_ball_columns = [
        column

        for column
        in required_ball_columns

        if column
        not in
        ball.columns
    ]


    if missing_ball_columns:

        raise ValueError(
            "Ball CSV missing required columns: "
            +
            ", ".join(
                missing_ball_columns
            )
        )


    ball[
        "frame"
    ] = pd.to_numeric(
        ball[
            "frame"
        ],
        errors="coerce",
    )


    ball = (
        ball
        .dropna(
            subset=[
                "frame"
            ]
        )
        .copy()
    )


    ball[
        "frame"
    ] = (
        ball[
            "frame"
        ]
        .astype(int)
    )


    ball[
        "ball_usable_bool"
    ] = (
        ball[
            "ball_usable"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    )


    for column in [
        "pitch_x_trusted_m",
        "pitch_y_trusted_m",
        "image_x_trusted",
        "image_y_trusted",
        "trusted_speed_mps",
    ]:

        ball[
            column
        ] = pd.to_numeric(
            ball[
                column
            ],
            errors="coerce",
        )


    # ========================================================
    # Frame lookup
    # ========================================================

    players_by_frame = {

        int(frame):
            group.copy()

        for (
            frame,
            group,
        )
        in players.groupby(
            "frame"
        )
    }


    ball_by_frame = (
        ball
        .set_index(
            "frame"
        )
    )


    if (
        len(players) == 0
        or
        len(ball) == 0
    ):

        raise RuntimeError(
            "Player or ball data became "
            "empty after normalization."
        )


    min_frame = max(
        int(
            players[
                "frame"
            ].min()
        ),
        int(
            ball[
                "frame"
            ].min()
        ),
    )


    max_frame = min(
        int(
            players[
                "frame"
            ].max()
        ),
        int(
            ball[
                "frame"
            ].max()
        ),
    )


    if max_frame < min_frame:

        raise RuntimeError(
            "Player and ball frame "
            "ranges do not overlap."
        )


    all_frames = range(
        min_frame,
        max_frame + 1,
    )


    rows = []

    projection_mismatch_count = 0


    # ========================================================
    # Per-frame geometric evidence
    # ========================================================

    for frame in all_frames:

        time_sec = (
            frame
            /
            fps
        )


        frame_players = (
            players_by_frame.get(
                frame,
                None,
            )
        )


        # ----------------------------------------------------
        # Player frame quality
        # ----------------------------------------------------

        if frame_players is None:

            A_count = 0
            B_count = 0

        else:

            A_count = int(
                (
                    frame_players[
                        "team"
                    ]
                    ==
                    "A"
                )
                .sum()
            )


            B_count = int(
                (
                    frame_players[
                        "team"
                    ]
                    ==
                    "B"
                )
                .sum()
            )


        player_frame_good = (

            A_count
            >=
            MIN_PLAYERS_PER_TEAM

            and

            B_count
            >=
            MIN_PLAYERS_PER_TEAM
        )


        # ----------------------------------------------------
        # Ball
        # ----------------------------------------------------

        if (
            frame
            not in
            ball_by_frame.index
        ):

            ball_usable = False
            ball_row = None

        else:

            ball_row = (
                ball_by_frame.loc[
                    frame
                ]
            )


            if isinstance(
                ball_row,
                pd.DataFrame,
            ):

                ball_row = (
                    ball_row.iloc[
                        0
                    ]
                )


            ball_usable = bool(
                ball_row[
                    "ball_usable_bool"
                ]
            )


        # ----------------------------------------------------
        # Default row
        # ----------------------------------------------------

        output = {
            "frame":
                frame,

            "time_sec":
                time_sec,

            "A_player_count":
                A_count,

            "B_player_count":
                B_count,

            "player_frame_good":
                player_frame_good,

            "ball_usable":
                ball_usable,

            "ball_quality":
                None,

            "ball_speed_mps":
                np.nan,

            "ball_pitch_x_m":
                np.nan,

            "ball_pitch_y_m":
                np.nan,

            "ball_image_x":
                np.nan,

            "ball_image_y":
                np.nan,

            "nearest_A_id":
                np.nan,

            "nearest_A_pitch_distance_m":
                np.nan,

            "nearest_A_image_foot_distance_px":
                np.nan,

            "nearest_B_id":
                np.nan,

            "nearest_B_pitch_distance_m":
                np.nan,

            "nearest_B_image_foot_distance_px":
                np.nan,

            "controller_team":
                None,

            "controller_id":
                np.nan,

            "direct_state":
                "Unknown",

            "projection_mismatch":
                False,
        }


        if not player_frame_good:

            output[
                "direct_state"
            ] = (
                "PlayerQualityLow"
            )

            rows.append(
                output
            )

            continue


        if (
            not ball_usable
            or
            ball_row is None
        ):

            output[
                "direct_state"
            ] = (
                "BallMissing"
            )

            rows.append(
                output
            )

            continue


        ball_pitch_x = (
            ball_row[
                "pitch_x_trusted_m"
            ]
        )


        ball_pitch_y = (
            ball_row[
                "pitch_y_trusted_m"
            ]
        )


        ball_image_x = (
            ball_row[
                "image_x_trusted"
            ]
        )


        ball_image_y = (
            ball_row[
                "image_y_trusted"
            ]
        )


        if (
            pd.isna(
                ball_pitch_x
            )
            or
            pd.isna(
                ball_pitch_y
            )
            or
            pd.isna(
                ball_image_x
            )
            or
            pd.isna(
                ball_image_y
            )
        ):

            output[
                "direct_state"
            ] = (
                "BallMissing"
            )

            rows.append(
                output
            )

            continue


        output[
            "ball_quality"
        ] = (
            ball_row.get(
                "ball_quality",
                None,
            )
        )


        output[
            "ball_speed_mps"
        ] = (
            ball_row.get(
                "trusted_speed_mps",
                np.nan,
            )
        )


        output[
            "ball_pitch_x_m"
        ] = ball_pitch_x


        output[
            "ball_pitch_y_m"
        ] = ball_pitch_y


        output[
            "ball_image_x"
        ] = ball_image_x


        output[
            "ball_image_y"
        ] = ball_image_y


        # ----------------------------------------------------
        # Nearest players
        # ----------------------------------------------------

        nearest_A = nearest_player(
            frame_players,
            "A",
            ball_pitch_x,
            ball_pitch_y,
            ball_image_x,
            ball_image_y,
            player_x,
            player_y,
        )


        nearest_B = nearest_player(
            frame_players,
            "B",
            ball_pitch_x,
            ball_pitch_y,
            ball_image_x,
            ball_image_y,
            player_x,
            player_y,
        )


        output[
            "nearest_A_id"
        ] = (
            nearest_A[
                "track_id"
            ]
        )


        output[
            "nearest_A_pitch_distance_m"
        ] = (
            nearest_A[
                "pitch_distance_m"
            ]
        )


        output[
            "nearest_A_image_foot_distance_px"
        ] = (
            nearest_A[
                "image_foot_distance_px"
            ]
        )


        output[
            "nearest_B_id"
        ] = (
            nearest_B[
                "track_id"
            ]
        )


        output[
            "nearest_B_pitch_distance_m"
        ] = (
            nearest_B[
                "pitch_distance_m"
            ]
        )


        output[
            "nearest_B_image_foot_distance_px"
        ] = (
            nearest_B[
                "image_foot_distance_px"
            ]
        )


        A_control = (
            nearest_A[
                "direct_control"
            ]
        )


        B_control = (
            nearest_B[
                "direct_control"
            ]
        )


        # ----------------------------------------------------
        # Projection mismatch
        # ----------------------------------------------------

        pitch_close = False
        foot_far = True


        for nearest in [
            nearest_A,
            nearest_B,
        ]:

            pitch_distance = (
                nearest[
                    "pitch_distance_m"
                ]
            )


            image_distance = (
                nearest[
                    "image_foot_distance_px"
                ]
            )


            if (
                pd.notna(
                    pitch_distance
                )
                and
                pitch_distance
                <=
                CONTROL_RADIUS_M
            ):

                pitch_close = True


                if (
                    pd.notna(
                        image_distance
                    )
                    and
                    image_distance
                    <=
                    FOOT_CONTROL_MAX_PX
                ):

                    foot_far = False


        projection_mismatch = (

            pitch_close
            and
            foot_far
        )


        output[
            "projection_mismatch"
        ] = (
            projection_mismatch
        )


        if projection_mismatch:

            projection_mismatch_count += 1


        # ----------------------------------------------------
        # Direct evidence
        # ----------------------------------------------------

        if (
            A_control
            and
            B_control
        ):

            pitch_diff = abs(
                nearest_A[
                    "pitch_distance_m"
                ]
                -
                nearest_B[
                    "pitch_distance_m"
                ]
            )


            foot_diff = abs(
                nearest_A[
                    "image_foot_distance_px"
                ]
                -
                nearest_B[
                    "image_foot_distance_px"
                ]
            )


            if (
                pitch_diff
                <=
                CONTESTED_DISTANCE_MARGIN_M

                or

                foot_diff
                <=
                CONTESTED_FOOT_MARGIN_PX
            ):

                output[
                    "direct_state"
                ] = (
                    "Contested"
                )


            elif (
                nearest_A[
                    "image_foot_distance_px"
                ]
                <
                nearest_B[
                    "image_foot_distance_px"
                ]
            ):

                output[
                    "direct_state"
                ] = "A"


                output[
                    "controller_team"
                ] = "A"


                output[
                    "controller_id"
                ] = (
                    nearest_A[
                        "track_id"
                    ]
                )


            else:

                output[
                    "direct_state"
                ] = "B"


                output[
                    "controller_team"
                ] = "B"


                output[
                    "controller_id"
                ] = (
                    nearest_B[
                        "track_id"
                    ]
                )


        elif A_control:

            output[
                "direct_state"
            ] = "A"


            output[
                "controller_team"
            ] = "A"


            output[
                "controller_id"
            ] = (
                nearest_A[
                    "track_id"
                ]
            )


        elif B_control:

            output[
                "direct_state"
            ] = "B"


            output[
                "controller_team"
            ] = "B"


            output[
                "controller_id"
            ] = (
                nearest_B[
                    "track_id"
                ]
            )


        else:

            output[
                "direct_state"
            ] = (
                "InFlight"
            )


        rows.append(
            output
        )


    result = pd.DataFrame(
        rows
    )


    stats = {
        "stable_tracks":
            int(
                len(
                    stable_track_stats
                )
            ),

        "removed_inconsistent_rows":
            removed_inconsistent_rows,

        "projection_mismatch_frames":
            projection_mismatch_count,
    }


    return (
        result,
        stats,
    )


# ============================================================
# V4 helper
# ============================================================

def player_control_candidate(
    row,
    team,
):

    pitch_distance = (
        row[
            f"nearest_{team}_pitch_distance_m"
        ]
    )


    foot_distance = (
        row[
            f"nearest_{team}_image_foot_distance_px"
        ]
    )


    speed = (
        row[
            "ball_speed_mps"
        ]
    )


    if (
        pd.isna(
            pitch_distance
        )
        or
        pd.isna(
            foot_distance
        )
    ):

        return False


    spatial_ok = (

        pitch_distance
        <=
        CONTROL_RADIUS_M

        and

        foot_distance
        <=
        FOOT_CONTROL_MAX_PX
    )


    speed_ok = (

        pd.isna(
            speed
        )

        or

        speed
        <=
        RECEPTION_MAX_SPEED_MPS
    )


    return (

        spatial_ok
        and
        speed_ok
    )


def is_aerial_contest(
    row,
):

    A_pitch = (
        row[
            "nearest_A_pitch_distance_m"
        ]
    )


    B_pitch = (
        row[
            "nearest_B_pitch_distance_m"
        ]
    )


    A_foot = (
        row[
            "nearest_A_image_foot_distance_px"
        ]
    )


    B_foot = (
        row[
            "nearest_B_image_foot_distance_px"
        ]
    )


    speed = (
        row[
            "ball_speed_mps"
        ]
    )


    if any(
        pd.isna(
            value
        )
        for value
        in [
            A_pitch,
            B_pitch,
            A_foot,
            B_foot,
        ]
    ):

        return False


    both_near = (

        A_pitch
        <=
        AERIAL_CONTEST_PITCH_RADIUS_M

        and

        B_pitch
        <=
        AERIAL_CONTEST_PITCH_RADIUS_M

        and

        A_foot
        <=
        AERIAL_CONTEST_IMAGE_RADIUS_PX

        and

        B_foot
        <=
        AERIAL_CONTEST_IMAGE_RADIUS_PX
    )


    if not both_near:

        return False


    similar_distance = (

        abs(
            A_pitch
            -
            B_pitch
        )
        <=
        AERIAL_CONTEST_PITCH_MARGIN_M

        or

        abs(
            A_foot
            -
            B_foot
        )
        <=
        AERIAL_CONTEST_IMAGE_MARGIN_PX
    )


    ball_is_aerial_or_fast = (

        row[
            "projection_mismatch_bool"
        ]

        or

        (
            pd.notna(
                speed
            )

            and

            speed
            >=
            AERIAL_CONTEST_MIN_SPEED_MPS
        )
    )


    return (

        similar_distance
        and
        ball_is_aerial_or_fast
    )


# ============================================================
# PASS B
#
# Possession V4:
#   physical evidence
#   +
#   reception confirmation
#   +
#   team possession
# ============================================================

def apply_v4_reasoning(
    seed,
):

    df = (
        seed
        .copy()
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )


    df[
        "ball_usable_bool"
    ] = normalize_bool(
        df[
            "ball_usable"
        ]
    )


    df[
        "player_frame_good_bool"
    ] = normalize_bool(
        df[
            "player_frame_good"
        ]
    )


    df[
        "projection_mismatch_bool"
    ] = normalize_bool(
        df[
            "projection_mismatch"
        ]
    )


    numeric_columns = [
        "frame",
        "time_sec",
        "ball_speed_mps",
        "nearest_A_id",
        "nearest_B_id",
        "nearest_A_pitch_distance_m",
        "nearest_B_pitch_distance_m",
        "nearest_A_image_foot_distance_px",
        "nearest_B_image_foot_distance_px",
    ]


    for column in numeric_columns:

        if column in df.columns:

            df[
                column
            ] = pd.to_numeric(
                df[
                    column
                ],
                errors="coerce",
            )


    # ========================================================
    # V4 PASS 1
    #
    # Physical evidence per frame.
    # ========================================================

    raw_states = []

    candidate_teams = []
    candidate_ids = []


    for _, row in df.iterrows():

        if not row[
            "ball_usable_bool"
        ]:

            raw_states.append(
                "Missing"
            )

            candidate_teams.append(
                None
            )

            candidate_ids.append(
                np.nan
            )

            continue


        if not row[
            "player_frame_good_bool"
        ]:

            raw_states.append(
                "PlayerQualityLow"
            )

            candidate_teams.append(
                None
            )

            candidate_ids.append(
                np.nan
            )

            continue


        speed = (
            row[
                "ball_speed_mps"
            ]
        )


        if is_aerial_contest(
            row
        ):

            raw_states.append(
                "AerialContest"
            )

            candidate_teams.append(
                None
            )

            candidate_ids.append(
                np.nan
            )

            continue


        if (
            pd.notna(
                speed
            )
            and
            speed
            >
            IN_TRANSIT_SPEED_MPS
        ):

            raw_states.append(
                "InTransit"
            )

            candidate_teams.append(
                None
            )

            candidate_ids.append(
                np.nan
            )

            continue


        A_candidate = (
            player_control_candidate(
                row,
                "A",
            )
        )


        B_candidate = (
            player_control_candidate(
                row,
                "B",
            )
        )


        if (
            A_candidate
            and
            B_candidate
        ):

            raw_states.append(
                "Contested"
            )

            candidate_teams.append(
                None
            )

            candidate_ids.append(
                np.nan
            )

            continue


        if A_candidate:

            raw_states.append(
                "ReceptionCandidate"
            )

            candidate_teams.append(
                "A"
            )

            candidate_ids.append(
                row[
                    "nearest_A_id"
                ]
            )

            continue


        if B_candidate:

            raw_states.append(
                "ReceptionCandidate"
            )

            candidate_teams.append(
                "B"
            )

            candidate_ids.append(
                row[
                    "nearest_B_id"
                ]
            )

            continue


        raw_states.append(
            "Loose"
        )

        candidate_teams.append(
            None
        )

        candidate_ids.append(
            np.nan
        )


    df[
        "raw_ball_state"
    ] = raw_states


    df[
        "reception_candidate_team"
    ] = candidate_teams


    df[
        "reception_candidate_id"
    ] = candidate_ids


    # ========================================================
    # V4 PASS 2
    #
    # Reception confirmation + team possession.
    # ========================================================

    df[
        "ball_state_v4"
    ] = "Unknown"


    df[
        "controller_team_v4"
    ] = None


    df[
        "controller_id_v4"
    ] = np.nan


    df[
        "possession_v4"
    ] = "Unknown"


    df[
        "possession_source_v4"
    ] = "unknown"


    df[
        "reception_candidate_count"
    ] = 0


    current_team = None


    reception_team = None
    reception_id = None
    reception_count = 0


    transit_streak = 0
    loose_streak = 0
    missing_streak = 0
    quality_streak = 0


    confirmed_receptions = 0
    confirmed_turnovers = 0
    aerial_contest_frames = 0


    for i in range(
        len(df)
    ):

        raw_state = (
            df.loc[
                i,
                "raw_ball_state"
            ]
        )


        # ----------------------------------------------------
        # Reception candidate
        # ----------------------------------------------------

        if (
            raw_state
            ==
            "ReceptionCandidate"
        ):

            transit_streak = 0
            loose_streak = 0
            missing_streak = 0
            quality_streak = 0


            team = (
                df.loc[
                    i,
                    "reception_candidate_team"
                ]
            )


            player_id = (
                df.loc[
                    i,
                    "reception_candidate_id"
                ]
            )


            same_candidate = (

                reception_team
                ==
                team

                and

                reception_id
                is not None

                and

                pd.notna(
                    player_id
                )

                and

                abs(
                    float(
                        reception_id
                    )
                    -
                    float(
                        player_id
                    )
                )
                <
                0.1
            )


            if same_candidate:

                reception_count += 1

            else:

                reception_team = team


                reception_id = (

                    float(
                        player_id
                    )

                    if
                    pd.notna(
                        player_id
                    )

                    else
                    None
                )


                reception_count = 1


            df.loc[
                i,
                "reception_candidate_count"
            ] = reception_count


            if (
                reception_count
                >=
                RECEPTION_CONFIRM_FRAMES
            ):

                new_team = team

                new_player = (
                    player_id
                )


                turnover = (

                    current_team
                    is not None

                    and

                    current_team
                    !=
                    new_team
                )


                if turnover:

                    confirmed_turnovers += 1


                if (
                    reception_count
                    ==
                    RECEPTION_CONFIRM_FRAMES
                ):

                    confirmed_receptions += 1


                current_team = (
                    new_team
                )


                df.loc[
                    i,
                    "ball_state_v4"
                ] = (
                    "Controlled"
                )


                df.loc[
                    i,
                    "controller_team_v4"
                ] = (
                    new_team
                )


                df.loc[
                    i,
                    "controller_id_v4"
                ] = (
                    new_player
                )


                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    new_team
                )


                if turnover:

                    df.loc[
                        i,
                        "possession_source_v4"
                    ] = (
                        "confirmed_"
                        "controlled_turnover"
                    )


                elif (
                    reception_count
                    ==
                    RECEPTION_CONFIRM_FRAMES
                ):

                    df.loc[
                        i,
                        "possession_source_v4"
                    ] = (
                        "confirmed_reception"
                    )


                else:

                    df.loc[
                        i,
                        "possession_source_v4"
                    ] = (
                        "controlled"
                    )


            else:

                df.loc[
                    i,
                    "ball_state_v4"
                ] = (
                    "Loose"
                )


                df.loc[
                    i,
                    "controller_team_v4"
                ] = None


                df.loc[
                    i,
                    "controller_id_v4"
                ] = np.nan


                if (
                    current_team
                    is not None
                ):

                    df.loc[
                        i,
                        "possession_v4"
                    ] = (
                        current_team
                    )


                    df.loc[
                        i,
                        "possession_source_v4"
                    ] = (
                        "pending_reception"
                    )


                else:

                    df.loc[
                        i,
                        "possession_v4"
                    ] = (
                        "Unknown"
                    )


                    df.loc[
                        i,
                        "possession_source_v4"
                    ] = (
                        "pending_initial_reception"
                    )


        # ----------------------------------------------------
        # In transit
        # ----------------------------------------------------

        elif (
            raw_state
            ==
            "InTransit"
        ):

            reception_team = None
            reception_id = None
            reception_count = 0

            loose_streak = 0
            missing_streak = 0
            quality_streak = 0

            transit_streak += 1


            df.loc[
                i,
                "ball_state_v4"
            ] = (
                "InTransit"
            )


            if (
                current_team
                is not None

                and

                transit_streak
                <=
                MAX_IN_TRANSIT_HOLD_FRAMES
            ):

                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    current_team
                )


                df.loc[
                    i,
                    "possession_source_v4"
                ] = (
                    "carried_in_transit"
                )


            else:

                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    "Unknown"
                )


                df.loc[
                    i,
                    "possession_source_v4"
                ] = (
                    "transit_too_long"
                )


        # ----------------------------------------------------
        # Aerial contest
        # ----------------------------------------------------

        elif (
            raw_state
            ==
            "AerialContest"
        ):

            reception_team = None
            reception_id = None
            reception_count = 0

            transit_streak = 0
            loose_streak = 0
            missing_streak = 0
            quality_streak = 0

            aerial_contest_frames += 1


            df.loc[
                i,
                "ball_state_v4"
            ] = (
                "AerialContest"
            )


            df.loc[
                i,
                "possession_v4"
            ] = (
                "Contested"
            )


            df.loc[
                i,
                "possession_source_v4"
            ] = (
                "aerial_contest"
            )


        # ----------------------------------------------------
        # Ground contest
        # ----------------------------------------------------

        elif (
            raw_state
            ==
            "Contested"
        ):

            reception_team = None
            reception_id = None
            reception_count = 0

            transit_streak = 0
            loose_streak = 0
            missing_streak = 0
            quality_streak = 0


            df.loc[
                i,
                "ball_state_v4"
            ] = (
                "Contested"
            )


            df.loc[
                i,
                "possession_v4"
            ] = (
                "Contested"
            )


            df.loc[
                i,
                "possession_source_v4"
            ] = (
                "spatial_contest"
            )


        # ----------------------------------------------------
        # Loose
        # ----------------------------------------------------

        elif (
            raw_state
            ==
            "Loose"
        ):

            reception_team = None
            reception_id = None
            reception_count = 0

            transit_streak = 0
            missing_streak = 0
            quality_streak = 0

            loose_streak += 1


            df.loc[
                i,
                "ball_state_v4"
            ] = (
                "Loose"
            )


            if (
                current_team
                is not None

                and

                loose_streak
                <=
                LOOSE_HOLD_FRAMES
            ):

                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    current_team
                )


                df.loc[
                    i,
                    "possession_source_v4"
                ] = (
                    "carried_loose"
                )


            else:

                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    "Unknown"
                )


                df.loc[
                    i,
                    "possession_source_v4"
                ] = (
                    "loose_too_long"
                )


        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------

        elif (
            raw_state
            ==
            "Missing"
        ):

            reception_team = None
            reception_id = None
            reception_count = 0

            transit_streak = 0
            loose_streak = 0
            quality_streak = 0

            missing_streak += 1


            df.loc[
                i,
                "ball_state_v4"
            ] = (
                "Missing"
            )


            if (
                current_team
                is not None

                and

                missing_streak
                <=
                BALL_MISSING_HOLD_FRAMES
            ):

                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    current_team
                )


                df.loc[
                    i,
                    "possession_source_v4"
                ] = (
                    "carried_missing"
                )


            else:

                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    "Unknown"
                )


                df.loc[
                    i,
                    "possession_source_v4"
                ] = (
                    "missing_too_long"
                )


        # ----------------------------------------------------
        # Low player quality
        # ----------------------------------------------------

        elif (
            raw_state
            ==
            "PlayerQualityLow"
        ):

            reception_team = None
            reception_id = None
            reception_count = 0

            transit_streak = 0
            loose_streak = 0
            missing_streak = 0

            quality_streak += 1


            df.loc[
                i,
                "ball_state_v4"
            ] = (
                "Unknown"
            )


            if (
                current_team
                is not None

                and

                quality_streak
                <=
                PLAYER_QUALITY_HOLD_FRAMES
            ):

                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    current_team
                )


                df.loc[
                    i,
                    "possession_source_v4"
                ] = (
                    "carried_low_player_quality"
                )


            else:

                df.loc[
                    i,
                    "possession_v4"
                ] = (
                    "Unknown"
                )


                df.loc[
                    i,
                    "possession_source_v4"
                ] = (
                    "player_quality_too_low"
                )


        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        else:

            reception_team = None
            reception_id = None
            reception_count = 0

            transit_streak = 0
            loose_streak = 0
            missing_streak = 0
            quality_streak = 0


            df.loc[
                i,
                "ball_state_v4"
            ] = (
                "Unknown"
            )


            df.loc[
                i,
                "possession_v4"
            ] = (
                "Unknown"
            )


            df.loc[
                i,
                "possession_source_v4"
            ] = (
                "unknown"
            )


    stats = {
        "confirmed_receptions":
            confirmed_receptions,

        "confirmed_turnovers":
            confirmed_turnovers,

        "aerial_contest_frames":
            aerial_contest_frames,
    }


    return (
        df,
        stats,
    )


# ============================================================
# Summary
# ============================================================

def build_summary(
    df,
    fps,
):

    A_frames = int(
        (
            df[
                "possession_v4"
            ]
            ==
            "A"
        ).sum()
    )


    B_frames = int(
        (
            df[
                "possession_v4"
            ]
            ==
            "B"
        ).sum()
    )


    contested_frames = int(
        (
            df[
                "possession_v4"
            ]
            ==
            "Contested"
        ).sum()
    )


    unknown_frames = int(
        (
            df[
                "possession_v4"
            ]
            ==
            "Unknown"
        ).sum()
    )


    known_team = (
        A_frames
        +
        B_frames
    )


    if known_team > 0:

        A_share = (
            A_frames
            /
            known_team
            *
            100.0
        )


        B_share = (
            B_frames
            /
            known_team
            *
            100.0
        )


    else:

        A_share = 0.0
        B_share = 0.0


    return pd.DataFrame(
        [
            {
                "state":
                    "A",

                "frames":
                    A_frames,

                "seconds":
                    A_frames
                    /
                    fps,

                "known_share_pct":
                    A_share,
            },

            {
                "state":
                    "B",

                "frames":
                    B_frames,

                "seconds":
                    B_frames
                    /
                    fps,

                "known_share_pct":
                    B_share,
            },

            {
                "state":
                    "Contested",

                "frames":
                    contested_frames,

                "seconds":
                    contested_frames
                    /
                    fps,

                "known_share_pct":
                    np.nan,
            },

            {
                "state":
                    "Unknown",

                "frames":
                    unknown_frames,

                "seconds":
                    unknown_frames
                    /
                    fps,

                "known_share_pct":
                    np.nan,
            },
        ]
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Canonical possession seed: "
            "V3 geometric evidence Pass 1 "
            "+ V4 physical-control reasoning."
        )
    )


    parser.add_argument(
        "--players",
        required=True,
    )


    parser.add_argument(
        "--ball",
        required=True,
    )


    parser.add_argument(
        "--output",
        required=True,
    )


    parser.add_argument(
        "--summary-output",
        default=None,
    )


    parser.add_argument(
        "--intermediate-output",
        default=None,
    )


    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
    )


    args = parser.parse_args()


    if args.fps <= 0:

        raise ValueError(
            "--fps must be > 0"
        )


    players_path = Path(
        args.players
    )


    ball_path = Path(
        args.ball
    )


    output_path = Path(
        args.output
    )


    summary_path = (

        Path(
            args.summary_output
        )

        if args.summary_output

        else

        output_path.parent
        /
        "possession_seed_summary.csv"
    )


    intermediate_path = (

        Path(
            args.intermediate_output
        )

        if args.intermediate_output

        else
        None
    )


    require_file(
        players_path,
        "Player tracking CSV",
    )


    require_file(
        ball_path,
        "Trusted ball CSV",
    )


    ensure_parent(
        output_path
    )


    ensure_parent(
        summary_path
    )


    if (
        intermediate_path
        is not None
    ):

        ensure_parent(
            intermediate_path
        )


    print("")
    print(
        "=" * 58
    )
    print(
        "POSSESSION SEED CLI"
    )
    print(
        "=" * 58
    )


    print(
        "Players:",
        players_path,
    )


    print(
        "Ball:",
        ball_path,
    )


    print(
        "Output:",
        output_path,
    )


    print(
        "Summary:",
        summary_path,
    )


    print(
        "FPS:",
        f"{args.fps:.3f}",
    )


    players = pd.read_csv(
        players_path
    )


    ball = pd.read_csv(
        ball_path
    )


    # ========================================================
    # PASS A
    # ========================================================

    print("")
    print(
        "=" * 58
    )
    print(
        "PASS A — V3 GEOMETRIC EVIDENCE"
    )
    print(
        "=" * 58
    )


    (
        seed,
        seed_stats,

    ) = build_geometric_seed(
        players=players,
        ball=ball,
        fps=args.fps,
    )


    if (
        intermediate_path
        is not None
    ):

        seed.to_csv(
            intermediate_path,
            index=False,
        )


        print(
            "Intermediate CSV:",
            intermediate_path,
        )


    print(
        "Frames:",
        len(seed),
    )


    print(
        "Stable tracks:",
        seed_stats[
            "stable_tracks"
        ],
    )


    print(
        "Removed team-inconsistent rows:",
        seed_stats[
            "removed_inconsistent_rows"
        ],
    )


    print(
        "Projection mismatch frames:",
        seed_stats[
            "projection_mismatch_frames"
        ],
    )


    print("")
    print(
        "Direct states:"
    )


    print(
        seed[
            "direct_state"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # PASS B
    # ========================================================

    print("")
    print(
        "=" * 58
    )
    print(
        "PASS B — V4 PHYSICAL CONTROL"
    )
    print(
        "=" * 58
    )


    (
        result,
        v4_stats,

    ) = apply_v4_reasoning(
        seed
    )


    result.to_csv(
        output_path,
        index=False,
    )


    summary = build_summary(
        result,
        args.fps,
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    print(
        "Frames:",
        len(result),
    )


    print("")
    print(
        "RAW BALL STATE"
    )


    print(
        result[
            "raw_ball_state"
        ]
        .value_counts()
        .to_string()
    )


    print("")
    print(
        "FINAL BALL STATE"
    )


    print(
        result[
            "ball_state_v4"
        ]
        .value_counts()
        .to_string()
    )


    print("")
    print(
        "FINAL POSSESSION"
    )


    print(
        result[
            "possession_v4"
        ]
        .value_counts()
        .to_string()
    )


    print("")


    print(
        "Confirmed receptions:",
        v4_stats[
            "confirmed_receptions"
        ],
    )


    print(
        "Confirmed controlled turnovers:",
        v4_stats[
            "confirmed_turnovers"
        ],
    )


    print(
        "Aerial contest frames:",
        v4_stats[
            "aerial_contest_frames"
        ],
    )


    print("")
    print(
        "POSSESSION SOURCE"
    )


    print(
        result[
            "possession_source_v4"
        ]
        .value_counts()
        .to_string()
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
        "Frames CSV:",
        output_path,
    )


    print(
        "Summary CSV:",
        summary_path,
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
