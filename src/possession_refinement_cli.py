#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Validated V5.3 parameter
# ============================================================

CONTEST_POST_LOCK_FRAMES = 4


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


def clean_text(
    value,
    default=None,
):
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


def estimate_fps(df):
    fps = 25.0

    if (
        "time_sec"
        not in
        df.columns
    ):
        return fps

    timing = (
        df[
            [
                "frame",
                "time_sec",
            ]
        ]
        .dropna()
        .sort_values(
            "frame"
        )
    )

    if len(timing) < 2:
        return fps

    frame_diff = (
        timing[
            "frame"
        ]
        .diff()
    )

    time_diff = (
        timing[
            "time_sec"
        ]
        .diff()
    )

    valid = (
        frame_diff.notna()
        &
        time_diff.notna()
        &
        (
            frame_diff
            >
            0
        )
        &
        (
            time_diff
            >
            0
        )
    )

    if valid.any():

        values = (
            frame_diff[
                valid
            ]
            /
            time_diff[
                valid
            ]
        )

        values = values[
            np.isfinite(
                values
            )
        ]

        if len(values) > 0:

            fps = float(
                values.median()
            )

    return fps


def build_summary(
    df,
    possession_column,
    ball_state_column,
    fps,
):
    rows = []

    for (
        key,
        value,

    ) in (
        df[
            possession_column
        ]
        .value_counts()
        .items()
    ):

        rows.append(
            {
                "type":
                    "possession",

                "name":
                    key,

                "frames":
                    int(value),

                "seconds":
                    float(value)
                    /
                    fps,
            }
        )

    for (
        key,
        value,

    ) in (
        df[
            ball_state_column
        ]
        .value_counts()
        .items()
    ):

        rows.append(
            {
                "type":
                    "ball_state",

                "name":
                    key,

                "frames":
                    int(value),

                "seconds":
                    float(value)
                    /
                    fps,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# V5.2 input normalization
# ============================================================

def normalize_v5_1(
    possession,
):
    possession = (
        possession.copy()
    )

    numeric_columns = [
        "frame",
        "time_sec",

        "controller_id_v4",
        "controller_id_v5",

        "last_touch_id_v5",

        "ball_speed_mps",
    ]

    for column in (
        numeric_columns
    ):

        if column in possession.columns:

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

    possession = (
        possession
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )

    return possession


def normalize_gate(
    gate,
):
    gate = (
        gate.copy()
    )

    numeric_columns = [
        "frame",
        "time_sec",

        "controller_id",

        "controller_run_id",
        "controller_run_length",
        "controller_run_position",

        "ball_speed_mps",

        "bbox_height_px",
        "foot_distance_px",
        "foot_distance_norm",
        "ball_rel_y",
    ]

    for column in (
        numeric_columns
    ):

        if column in gate.columns:

            gate[
                column
            ] = pd.to_numeric(
                gate[
                    column
                ],
                errors="coerce",
            )

    gate = (
        gate
        .dropna(
            subset=[
                "frame"
            ]
        )
        .copy()
    )

    gate[
        "frame"
    ] = (
        gate[
            "frame"
        ]
        .astype(int)
    )

    gate = (
        gate
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )

    return gate


# ============================================================
# PASS A
#
# Possession V5.2
# Controller Gate V2 integration
# ============================================================

def run_v5_2(
    possession,
    gate,
):
    possession = normalize_v5_1(
        possession
    )

    gate = normalize_gate(
        gate
    )

    fps = estimate_fps(
        possession
    )

    gate_by_frame = (
        gate
        .set_index(
            "frame"
        )
    )

    rows = []
    audit_rows = []

    for (
        _,
        base,

    ) in possession.iterrows():

        frame = int(
            base[
                "frame"
            ]
        )

        row = (
            base.to_dict()
        )

        # ====================================================
        # V5.1 baseline
        # ====================================================

        possession_v5 = clean_text(
            base.get(
                "possession_v5",
                None,
            ),
            "Unknown",
        )

        ball_state_v5 = clean_text(
            base.get(
                "ball_state_v5",
                None,
            ),
            "Unknown",
        )

        controller_team_v5 = clean_team(
            base.get(
                "controller_team_v5",
                None,
            )
        )

        controller_id_v5 = clean_id(
            base.get(
                "controller_id_v5",
                np.nan,
            )
        )

        last_touch_team_v5 = clean_team(
            base.get(
                "last_touch_team_v5",
                None,
            )
        )

        last_touch_id_v5 = clean_id(
            base.get(
                "last_touch_id_v5",
                np.nan,
            )
        )

        last_touch_type_v5 = clean_text(
            base.get(
                "last_touch_type_v5",
                None,
            ),
            None,
        )

        source_v5 = clean_text(
            base.get(
                "source_v5",
                None,
            ),
            "baseline_v4",
        )

        # ====================================================
        # Default V5.2 = V5.1
        # ====================================================

        possession_v5_2 = (
            possession_v5
        )

        ball_state_v5_2 = (
            ball_state_v5
        )

        controller_team_v5_2 = (
            controller_team_v5
        )

        controller_id_v5_2 = (
            controller_id_v5
        )

        last_touch_team_v5_2 = (
            last_touch_team_v5
        )

        last_touch_id_v5_2 = (
            last_touch_id_v5
        )

        last_touch_type_v5_2 = (
            last_touch_type_v5
        )

        source_v5_2 = (
            source_v5
        )

        controller_gate_status = None
        controller_gate_reason = None
        controller_gate_mode = None

        controller_gate_override = False

        # ====================================================
        # Controller gate
        # ====================================================

        gate_row = None

        if (
            frame
            in
            gate_by_frame.index
        ):

            gate_row = (
                gate_by_frame.loc[
                    frame
                ]
            )

            if isinstance(
                gate_row,
                pd.DataFrame,
            ):

                gate_row = (
                    gate_row.iloc[
                        0
                    ]
                )

            controller_gate_status = clean_text(
                gate_row.get(
                    "gate_status",
                    None,
                ),
                None,
            )

            controller_gate_reason = clean_text(
                gate_row.get(
                    "gate_reason",
                    None,
                ),
                None,
            )

            controller_gate_mode = clean_text(
                gate_row.get(
                    "control_mode",
                    None,
                ),
                None,
            )

        # ====================================================
        # Only REJECT overrides V5.1
        # ====================================================

        if (
            controller_gate_status
            ==
            "REJECT"

            and

            ball_state_v5
            ==
            "Controlled"
        ):

            controller_gate_override = True

            # ------------------------------------------------
            # Contest lock
            # ------------------------------------------------

            if (
                controller_gate_reason
                ==
                "contest_lock"
            ):

                possession_v5_2 = (
                    "Contested"
                )

                ball_state_v5_2 = (
                    "AerialContest"
                )

                controller_team_v5_2 = None
                controller_id_v5_2 = np.nan

                last_touch_team_v5_2 = None
                last_touch_id_v5_2 = np.nan

                last_touch_type_v5_2 = (
                    "ContestUnknown"
                )

                source_v5_2 = (
                    "controller_gate_contest_lock"
                )

            # ------------------------------------------------
            # Hard geometry rejection
            # ------------------------------------------------

            else:

                possession_v5_2 = (
                    possession_v5
                )

                ball_state_v5_2 = (
                    "Uncontrolled"
                )

                controller_team_v5_2 = None
                controller_id_v5_2 = np.nan

                rejected_same_as_last_touch = (
                    last_touch_type_v5
                    ==
                    "Controlled"

                    and

                    clean_team(
                        last_touch_team_v5
                    )
                    ==
                    clean_team(
                        controller_team_v5
                    )

                    and

                    same_player_id(
                        last_touch_id_v5,
                        controller_id_v5,
                    )
                )

                if (
                    rejected_same_as_last_touch
                ):

                    last_touch_team_v5_2 = None
                    last_touch_id_v5_2 = np.nan
                    last_touch_type_v5_2 = None

                source_v5_2 = (
                    "controller_gate_reject_"
                    +
                    str(
                        controller_gate_reason
                    )
                )

        # ====================================================
        # Save V5.2
        # ====================================================

        row[
            "possession_v5_2"
        ] = possession_v5_2

        row[
            "ball_state_v5_2"
        ] = ball_state_v5_2

        row[
            "controller_team_v5_2"
        ] = controller_team_v5_2

        row[
            "controller_id_v5_2"
        ] = controller_id_v5_2

        row[
            "last_touch_team_v5_2"
        ] = last_touch_team_v5_2

        row[
            "last_touch_id_v5_2"
        ] = last_touch_id_v5_2

        row[
            "last_touch_type_v5_2"
        ] = last_touch_type_v5_2

        row[
            "source_v5_2"
        ] = source_v5_2

        row[
            "controller_gate_status"
        ] = controller_gate_status

        row[
            "controller_gate_reason"
        ] = controller_gate_reason

        row[
            "controller_gate_mode"
        ] = controller_gate_mode

        row[
            "controller_gate_override"
        ] = controller_gate_override

        # ====================================================
        # Gate QA metrics
        # ====================================================

        if (
            gate_row
            is not None
        ):

            row[
                "controller_gate_run_length"
            ] = gate_row.get(
                "controller_run_length",
                np.nan,
            )

            row[
                "controller_gate_run_position"
            ] = gate_row.get(
                "controller_run_position",
                np.nan,
            )

            row[
                "controller_gate_foot_distance_px"
            ] = gate_row.get(
                "foot_distance_px",
                np.nan,
            )

            row[
                "controller_gate_foot_distance_norm"
            ] = gate_row.get(
                "foot_distance_norm",
                np.nan,
            )

            row[
                "controller_gate_ball_rel_y"
            ] = gate_row.get(
                "ball_rel_y",
                np.nan,
            )

        else:

            row[
                "controller_gate_run_length"
            ] = np.nan

            row[
                "controller_gate_run_position"
            ] = np.nan

            row[
                "controller_gate_foot_distance_px"
            ] = np.nan

            row[
                "controller_gate_foot_distance_norm"
            ] = np.nan

            row[
                "controller_gate_ball_rel_y"
            ] = np.nan

        rows.append(
            row
        )

        # ====================================================
        # Audit
        # ====================================================

        if (
            controller_gate_override
        ):

            audit_rows.append(
                {
                    "frame":
                        frame,

                    "time_sec":
                        base.get(
                            "time_sec",
                            frame / fps,
                        ),

                    "possession_v5":
                        possession_v5,

                    "possession_v5_2":
                        possession_v5_2,

                    "ball_state_v5":
                        ball_state_v5,

                    "ball_state_v5_2":
                        ball_state_v5_2,

                    "controller_team_v5":
                        controller_team_v5,

                    "controller_id_v5":
                        controller_id_v5,

                    "controller_team_v5_2":
                        controller_team_v5_2,

                    "controller_id_v5_2":
                        controller_id_v5_2,

                    "last_touch_team_v5":
                        last_touch_team_v5,

                    "last_touch_id_v5":
                        last_touch_id_v5,

                    "last_touch_team_v5_2":
                        last_touch_team_v5_2,

                    "last_touch_id_v5_2":
                        last_touch_id_v5_2,

                    "gate_status":
                        controller_gate_status,

                    "gate_reason":
                        controller_gate_reason,

                    "gate_mode":
                        controller_gate_mode,

                    "source_v5":
                        source_v5,

                    "source_v5_2":
                        source_v5_2,

                    "foot_distance_norm":
                        (
                            gate_row.get(
                                "foot_distance_norm",
                                np.nan,
                            )

                            if
                            gate_row
                            is not None

                            else
                            np.nan
                        ),

                    "ball_rel_y":
                        (
                            gate_row.get(
                                "ball_rel_y",
                                np.nan,
                            )

                            if
                            gate_row
                            is not None

                            else
                            np.nan
                        ),
                }
            )

    v5_2 = pd.DataFrame(
        rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    summary = build_summary(
        v5_2,
        "possession_v5_2",
        "ball_state_v5_2",
        fps,
    )

    return (
        v5_2,
        audit,
        summary,
        fps,
    )


# ============================================================
# V5.3 input normalization
# ============================================================

def normalize_v5_2(
    possession,
):
    possession = (
        possession.copy()
    )

    numeric_columns = [
        "frame",
        "time_sec",

        "controller_id_v5_2",
        "last_touch_id_v5_2",

        "ball_speed_mps",

        "controller_gate_run_length",
        "controller_gate_run_position",
    ]

    for column in (
        numeric_columns
    ):

        if column in possession.columns:

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

    possession = (
        possession
        .sort_values(
            "frame"
        )
        .reset_index(
            drop=True
        )
    )

    return possession


def normalize_events(
    events,
):
    events = (
        events.copy()
    )

    numeric_columns = [
        "physical_event_id",
        "start_frame",
        "end_frame",
        "last_touch_track_id",
    ]

    for column in (
        numeric_columns
    ):

        if column in events.columns:

            events[
                column
            ] = pd.to_numeric(
                events[
                    column
                ],
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

    events[
        "start_frame"
    ] = (
        events[
            "start_frame"
        ]
        .astype(int)
    )

    events[
        "end_frame"
    ] = (
        events[
            "end_frame"
        ]
        .astype(int)
    )

    events = (
        events
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

    return events


# ============================================================
# PASS B
#
# Possession V5.3
# Contest Episode / Resolution Gate
# ============================================================

def run_v5_3(
    possession,
    events,
):
    possession = normalize_v5_2(
        possession
    )

    events = normalize_events(
        events
    )

    fps = estimate_fps(
        possession
    )

    # ========================================================
    # AirTouch frames
    # ========================================================

    air_touch_frames = set()

    confirmed_air_touch_events = (
        events[
            events[
                "final_class"
            ]
            ==
            "AirTouch"
        ]
        .copy()
    )

    for (
        _,
        event,

    ) in confirmed_air_touch_events.iterrows():

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

            air_touch_frames.add(
                frame
            )

    # ========================================================
    # Build contest episodes
    # ========================================================

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

    episode_rows = []

    for (
        _,
        event,

    ) in physical_contests.iterrows():

        confirmed_start = int(
            event[
                "start_frame"
            ]
        )

        confirmed_end = int(
            event[
                "end_frame"
            ]
        )

        proposed_end = (
            confirmed_end
            +
            CONTEST_POST_LOCK_FRAMES
        )

        actual_end = (
            proposed_end
        )

        resolution_reason = (
            "minimum_contest_persistence_complete"
        )

        # ----------------------------------------------------
        # Early resolution by confirmed AirTouch
        # ----------------------------------------------------

        for frame in range(
            confirmed_start + 1,
            proposed_end + 1,
        ):

            if (
                frame
                in
                air_touch_frames
            ):

                actual_end = (
                    frame - 1
                )

                resolution_reason = (
                    "resolved_by_confirmed_air_touch"
                )

                break

        actual_end = max(
            confirmed_end,
            actual_end,
        )

        episode_rows.append(
            {
                "contest_event_id":
                    int(
                        event.get(
                            "physical_event_id",
                            len(
                                episode_rows
                            )
                            +
                            1,
                        )
                    ),

                "confirmed_start_frame":
                    confirmed_start,

                "confirmed_end_frame":
                    confirmed_end,

                "episode_start_frame":
                    confirmed_start,

                "episode_end_frame":
                    actual_end,

                "episode_start_time_sec":
                    confirmed_start
                    /
                    fps,

                "episode_end_time_sec":
                    actual_end
                    /
                    fps,

                "participants":
                    event.get(
                        "participants",
                        None,
                    ),

                "resolution_reason":
                    resolution_reason,
            }
        )

    contest_episodes = pd.DataFrame(
        episode_rows
    )

    # ========================================================
    # Frame → contest episode
    # ========================================================

    contest_episode_by_frame = {}

    for (
        _,
        episode,

    ) in contest_episodes.iterrows():

        start = int(
            episode[
                "episode_start_frame"
            ]
        )

        end = int(
            episode[
                "episode_end_frame"
            ]
        )

        for frame in range(
            start,
            end + 1,
        ):

            contest_episode_by_frame[
                frame
            ] = episode

    # ========================================================
    # Main pass
    # ========================================================

    rows = []
    audit_rows = []

    for (
        _,
        base,

    ) in possession.iterrows():

        frame = int(
            base[
                "frame"
            ]
        )

        row = (
            base.to_dict()
        )

        possession_v5_2 = clean_text(
            base.get(
                "possession_v5_2",
                None,
            ),
            "Unknown",
        )

        ball_state_v5_2 = clean_text(
            base.get(
                "ball_state_v5_2",
                None,
            ),
            "Unknown",
        )

        controller_team_v5_2 = clean_team(
            base.get(
                "controller_team_v5_2",
                None,
            )
        )

        controller_id_v5_2 = clean_id(
            base.get(
                "controller_id_v5_2",
                np.nan,
            )
        )

        last_touch_team_v5_2 = clean_team(
            base.get(
                "last_touch_team_v5_2",
                None,
            )
        )

        last_touch_id_v5_2 = clean_id(
            base.get(
                "last_touch_id_v5_2",
                np.nan,
            )
        )

        last_touch_type_v5_2 = clean_text(
            base.get(
                "last_touch_type_v5_2",
                None,
            ),
            None,
        )

        source_v5_2 = clean_text(
            base.get(
                "source_v5_2",
                None,
            ),
            "baseline_v4",
        )

        # ====================================================
        # Default V5.3 = V5.2
        # ====================================================

        possession_v5_3 = (
            possession_v5_2
        )

        ball_state_v5_3 = (
            ball_state_v5_2
        )

        controller_team_v5_3 = (
            controller_team_v5_2
        )

        controller_id_v5_3 = (
            controller_id_v5_2
        )

        last_touch_team_v5_3 = (
            last_touch_team_v5_2
        )

        last_touch_id_v5_3 = (
            last_touch_id_v5_2
        )

        last_touch_type_v5_3 = (
            last_touch_type_v5_2
        )

        source_v5_3 = (
            source_v5_2
        )

        contest_episode_active = False
        contest_episode_id = np.nan
        contest_episode_phase = None
        contest_episode_participants = None

        episode = (
            contest_episode_by_frame.get(
                frame,
                None,
            )
        )

        confirmed_air_touch_here = (
            frame
            in
            air_touch_frames
        )

        if (
            episode is not None

            and

            not confirmed_air_touch_here
        ):

            contest_episode_active = True

            contest_episode_id = (
                episode.get(
                    "contest_event_id",
                    np.nan,
                )
            )

            contest_episode_participants = (
                episode.get(
                    "participants",
                    None,
                )
            )

            confirmed_start = int(
                episode[
                    "confirmed_start_frame"
                ]
            )

            confirmed_end = int(
                episode[
                    "confirmed_end_frame"
                ]
            )

            if (
                frame
                >=
                confirmed_start

                and

                frame
                <=
                confirmed_end
            ):

                contest_episode_phase = (
                    "ContestConfirmed"
                )

                source_v5_3 = (
                    "contest_episode_confirmed"
                )

            else:

                contest_episode_phase = (
                    "ContestCarry"
                )

                source_v5_3 = (
                    "contest_episode_carry"
                )

            possession_v5_3 = (
                "Contested"
            )

            ball_state_v5_3 = (
                "AerialContest"
            )

            controller_team_v5_3 = None
            controller_id_v5_3 = np.nan

            last_touch_team_v5_3 = None
            last_touch_id_v5_3 = np.nan

            last_touch_type_v5_3 = (
                "AerialContestUnknown"
            )

        # ====================================================
        # Save V5.3
        # ====================================================

        row[
            "possession_v5_3"
        ] = possession_v5_3

        row[
            "ball_state_v5_3"
        ] = ball_state_v5_3

        row[
            "controller_team_v5_3"
        ] = controller_team_v5_3

        row[
            "controller_id_v5_3"
        ] = controller_id_v5_3

        row[
            "last_touch_team_v5_3"
        ] = last_touch_team_v5_3

        row[
            "last_touch_id_v5_3"
        ] = last_touch_id_v5_3

        row[
            "last_touch_type_v5_3"
        ] = last_touch_type_v5_3

        row[
            "source_v5_3"
        ] = source_v5_3

        row[
            "contest_episode_active"
        ] = contest_episode_active

        row[
            "contest_episode_id"
        ] = contest_episode_id

        row[
            "contest_episode_phase"
        ] = contest_episode_phase

        row[
            "contest_episode_participants"
        ] = contest_episode_participants

        changed_from_v5_2 = (
            str(
                possession_v5_3
            )
            !=
            str(
                possession_v5_2
            )

            or

            str(
                ball_state_v5_3
            )
            !=
            str(
                ball_state_v5_2
            )

            or

            clean_team(
                controller_team_v5_3
            )
            !=
            clean_team(
                controller_team_v5_2
            )

            or

            (
                pd.notna(
                    controller_id_v5_3
                )
                !=
                pd.notna(
                    controller_id_v5_2
                )
            )
        )

        row[
            "contest_changed_from_v5_2"
        ] = changed_from_v5_2

        rows.append(
            row
        )

        if (
            contest_episode_active
        ):

            audit_rows.append(
                {
                    "frame":
                        frame,

                    "time_sec":
                        base.get(
                            "time_sec",
                            frame / fps,
                        ),

                    "possession_v5_2":
                        possession_v5_2,

                    "possession_v5_3":
                        possession_v5_3,

                    "ball_state_v5_2":
                        ball_state_v5_2,

                    "ball_state_v5_3":
                        ball_state_v5_3,

                    "controller_team_v5_2":
                        controller_team_v5_2,

                    "controller_id_v5_2":
                        controller_id_v5_2,

                    "controller_team_v5_3":
                        controller_team_v5_3,

                    "controller_id_v5_3":
                        controller_id_v5_3,

                    "contest_episode_id":
                        contest_episode_id,

                    "contest_episode_phase":
                        contest_episode_phase,

                    "participants":
                        contest_episode_participants,

                    "changed_from_v5_2":
                        changed_from_v5_2,

                    "source_v5_2":
                        source_v5_2,

                    "source_v5_3":
                        source_v5_3,
                }
            )

    v5_3 = pd.DataFrame(
        rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    summary = build_summary(
        v5_3,
        "possession_v5_3",
        "ball_state_v5_3",
        fps,
    )

    return (
        v5_3,
        audit,
        summary,
        contest_episodes,
        fps,
    )


# ============================================================
# Regression helpers
# ============================================================

def get_frame(
    df,
    frame,
):
    rows = (
        df[
            df[
                "frame"
            ]
            ==
            frame
        ]
    )

    if len(rows) == 0:
        return None

    return (
        rows.iloc[0]
    )


def is_contest_frame(
    df,
    frame,
):
    row = get_frame(
        df,
        frame,
    )

    if row is None:
        return False

    return (
        str(
            row[
                "possession_v5_3"
            ]
        )
        ==
        "Contested"

        and

        str(
            row[
                "ball_state_v5_3"
            ]
        )
        ==
        "AerialContest"

        and

        clean_team(
            row[
                "controller_team_v5_3"
            ]
        )
        is None
    )


def run_v5_2_checks(
    v5_2,
):
    checks = {}

    r159 = get_frame(
        v5_2,
        159,
    )

    checks[
        "159_long_ball"
    ] = (
        r159 is not None
        and
        str(
            r159[
                "possession_v5_2"
            ]
        )
        ==
        "B"
        and
        str(
            r159[
                "ball_state_v5_2"
            ]
        )
        ==
        "InTransit"
        and
        clean_team(
            r159[
                "controller_team_v5_2"
            ]
        )
        is None
    )

    r172 = get_frame(
        v5_2,
        172,
    )

    checks[
        "172_aerial_contest"
    ] = (
        r172 is not None
        and
        str(
            r172[
                "possession_v5_2"
            ]
        )
        ==
        "Contested"
        and
        str(
            r172[
                "ball_state_v5_2"
            ]
        )
        ==
        "AerialContest"
        and
        clean_team(
            r172[
                "controller_team_v5_2"
            ]
        )
        is None
    )

    r175 = get_frame(
        v5_2,
        175,
    )

    checks[
        "175_contest_lock"
    ] = (
        r175 is not None
        and
        str(
            r175[
                "possession_v5_2"
            ]
        )
        ==
        "Contested"
        and
        str(
            r175[
                "ball_state_v5_2"
            ]
        )
        ==
        "AerialContest"
        and
        clean_team(
            r175[
                "controller_team_v5_2"
            ]
        )
        is None
    )

    r203 = get_frame(
        v5_2,
        203,
    )

    checks[
        "203_reception_bounce"
    ] = (
        r203 is not None
        and
        str(
            r203[
                "possession_v5_2"
            ]
        )
        ==
        str(
            r203[
                "possession_v5"
            ]
        )
        and
        str(
            r203[
                "ball_state_v5_2"
            ]
        )
        ==
        str(
            r203[
                "ball_state_v5"
            ]
        )
    )

    r226 = get_frame(
        v5_2,
        226,
    )

    checks[
        "226_false_control"
    ] = (
        r226 is not None
        and
        str(
            r226[
                "possession_v5_2"
            ]
        )
        ==
        "A"
        and
        str(
            r226[
                "ball_state_v5_2"
            ]
        )
        ==
        "Uncontrolled"
        and
        clean_team(
            r226[
                "controller_team_v5_2"
            ]
        )
        is None
    )

    r253 = get_frame(
        v5_2,
        253,
    )

    checks[
        "253_aerial_contest"
    ] = (
        r253 is not None
        and
        str(
            r253[
                "possession_v5_2"
            ]
        )
        ==
        "Contested"
        and
        str(
            r253[
                "ball_state_v5_2"
            ]
        )
        ==
        "AerialContest"
        and
        clean_team(
            r253[
                "controller_team_v5_2"
            ]
        )
        is None
    )

    r281 = get_frame(
        v5_2,
        281,
    )

    checks[
        "281_air_touch"
    ] = (
        r281 is not None
        and
        str(
            r281[
                "possession_v5_2"
            ]
        )
        ==
        "B"
        and
        str(
            r281[
                "ball_state_v5_2"
            ]
        )
        ==
        "InTransit"
        and
        clean_team(
            r281[
                "controller_team_v5_2"
            ]
        )
        is None
    )

    r287 = get_frame(
        v5_2,
        287,
    )

    checks[
        "287_post_contest"
    ] = (
        r287 is not None
        and
        str(
            r287[
                "possession_v5_2"
            ]
        )
        ==
        str(
            r287[
                "possession_v5"
            ]
        )
        and
        str(
            r287[
                "ball_state_v5_2"
            ]
        )
        ==
        str(
            r287[
                "ball_state_v5"
            ]
        )
    )

    r650 = get_frame(
        v5_2,
        650,
    )

    checks[
        "650_valid_dribble"
    ] = (
        r650 is not None
        and
        str(
            r650[
                "possession_v5_2"
            ]
        )
        ==
        "A"
        and
        str(
            r650[
                "ball_state_v5_2"
            ]
        )
        ==
        "Controlled"
        and
        clean_team(
            r650[
                "controller_team_v5_2"
            ]
        )
        ==
        "A"
        and
        same_player_id(
            r650[
                "controller_id_v5_2"
            ],
            11,
        )
    )

    return checks


def run_v5_3_checks(
    v5_3,
):
    checks = {}

    r159 = get_frame(
        v5_3,
        159,
    )

    checks[
        "159_long_ball"
    ] = (
        r159 is not None
        and
        str(
            r159[
                "possession_v5_3"
            ]
        )
        ==
        "B"
        and
        str(
            r159[
                "ball_state_v5_3"
            ]
        )
        ==
        "InTransit"
    )

    for frame in [
        172,
        173,
        174,
        175,
        176,
    ]:

        checks[
            f"{frame}_contest"
        ] = (
            is_contest_frame(
                v5_3,
                frame,
            )
        )

    r177 = get_frame(
        v5_3,
        177,
    )

    checks[
        "177_resolution"
    ] = (
        r177 is not None
        and
        str(
            r177[
                "possession_v5_3"
            ]
        )
        ==
        str(
            r177[
                "possession_v5_2"
            ]
        )
        and
        str(
            r177[
                "ball_state_v5_3"
            ]
        )
        ==
        str(
            r177[
                "ball_state_v5_2"
            ]
        )
        and
        bool(
            r177[
                "contest_episode_active"
            ]
        )
        is False
    )

    r203 = get_frame(
        v5_3,
        203,
    )

    checks[
        "203_reception_bounce"
    ] = (
        r203 is not None
        and
        str(
            r203[
                "possession_v5_3"
            ]
        )
        ==
        str(
            r203[
                "possession_v5_2"
            ]
        )
        and
        str(
            r203[
                "ball_state_v5_3"
            ]
        )
        ==
        str(
            r203[
                "ball_state_v5_2"
            ]
        )
    )

    r226 = get_frame(
        v5_3,
        226,
    )

    checks[
        "226_false_control"
    ] = (
        r226 is not None
        and
        str(
            r226[
                "possession_v5_3"
            ]
        )
        ==
        "A"
        and
        str(
            r226[
                "ball_state_v5_3"
            ]
        )
        ==
        "Uncontrolled"
        and
        clean_team(
            r226[
                "controller_team_v5_3"
            ]
        )
        is None
    )

    r253 = get_frame(
        v5_3,
        253,
    )

    checks[
        "253_aerial_contest"
    ] = (
        r253 is not None
        and
        str(
            r253[
                "possession_v5_3"
            ]
        )
        ==
        "Contested"
        and
        str(
            r253[
                "ball_state_v5_3"
            ]
        )
        ==
        "AerialContest"
        and
        clean_team(
            r253[
                "controller_team_v5_3"
            ]
        )
        is None
    )

    r281 = get_frame(
        v5_3,
        281,
    )

    checks[
        "281_air_touch"
    ] = (
        r281 is not None
        and
        str(
            r281[
                "possession_v5_3"
            ]
        )
        ==
        "B"
        and
        str(
            r281[
                "ball_state_v5_3"
            ]
        )
        ==
        "InTransit"
        and
        clean_team(
            r281[
                "controller_team_v5_3"
            ]
        )
        is None
        and
        clean_team(
            r281[
                "last_touch_team_v5_3"
            ]
        )
        ==
        "B"
        and
        same_player_id(
            r281[
                "last_touch_id_v5_3"
            ],
            10,
        )
    )

    r287 = get_frame(
        v5_3,
        287,
    )

    checks[
        "287_post_contest"
    ] = (
        r287 is not None
        and
        str(
            r287[
                "possession_v5_3"
            ]
        )
        ==
        str(
            r287[
                "possession_v5_2"
            ]
        )
        and
        str(
            r287[
                "ball_state_v5_3"
            ]
        )
        ==
        str(
            r287[
                "ball_state_v5_2"
            ]
        )
    )

    r650 = get_frame(
        v5_3,
        650,
    )

    checks[
        "650_valid_dribble"
    ] = (
        r650 is not None
        and
        str(
            r650[
                "possession_v5_3"
            ]
        )
        ==
        "A"
        and
        str(
            r650[
                "ball_state_v5_3"
            ]
        )
        ==
        "Controlled"
        and
        clean_team(
            r650[
                "controller_team_v5_3"
            ]
        )
        ==
        "A"
        and
        same_player_id(
            r650[
                "controller_id_v5_3"
            ],
            11,
        )
    )

    return checks


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Canonical possession refinement: "
            "V5.2 Controller Gate integration "
            "+ V5.3 contest persistence."
        )
    )

    parser.add_argument(
        "--possession",
        required=True,
        help=(
            "Canonical Possession V5.1 "
            "frames CSV"
        ),
    )

    parser.add_argument(
        "--gate",
        required=True,
        help=(
            "Canonical Controller Gate CSV"
        ),
    )

    parser.add_argument(
        "--events",
        required=True,
        help=(
            "Canonical Air Interaction "
            "physical events CSV"
        ),
    )

    parser.add_argument(
        "--intermediate-output",
        required=True,
        help=(
            "V5.2 intermediate frames CSV"
        ),
    )

    parser.add_argument(
        "--intermediate-summary-output",
        default=None,
    )

    parser.add_argument(
        "--intermediate-audit-output",
        default=None,
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Final V5.3 frames CSV"
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=None,
    )

    parser.add_argument(
        "--audit-output",
        default=None,
    )

    parser.add_argument(
        "--episodes-output",
        default=None,
    )

    args = parser.parse_args()

    possession_path = Path(
        args.possession
    )

    gate_path = Path(
        args.gate
    )

    events_path = Path(
        args.events
    )

    intermediate_path = Path(
        args.intermediate_output
    )

    output_path = Path(
        args.output
    )

    intermediate_summary_path = (
        Path(
            args.intermediate_summary_output
        )

        if
        args.intermediate_summary_output

        else

        intermediate_path.parent
        /
        "possession_v5_2_summary.csv"
    )

    intermediate_audit_path = (
        Path(
            args.intermediate_audit_output
        )

        if
        args.intermediate_audit_output

        else

        intermediate_path.parent
        /
        "possession_v5_2_audit.csv"
    )

    summary_path = (
        Path(
            args.summary_output
        )

        if
        args.summary_output

        else

        output_path.parent
        /
        "possession_v5_3_summary.csv"
    )

    audit_path = (
        Path(
            args.audit_output
        )

        if
        args.audit_output

        else

        output_path.parent
        /
        "possession_v5_3_audit.csv"
    )

    episodes_path = (
        Path(
            args.episodes_output
        )

        if
        args.episodes_output

        else

        output_path.parent
        /
        "possession_v5_3_contest_episodes.csv"
    )

    require_file(
        possession_path,
        "Possession V5.1 CSV",
    )

    require_file(
        gate_path,
        "Controller Gate CSV",
    )

    require_file(
        events_path,
        "Air Interaction events CSV",
    )

    for path in [
        intermediate_path,
        intermediate_summary_path,
        intermediate_audit_path,
        output_path,
        summary_path,
        audit_path,
        episodes_path,
    ]:

        ensure_parent(
            path
        )

    print("")
    print(
        "=" * 62
    )
    print(
        "POSSESSION REFINEMENT CLI"
    )
    print(
        "=" * 62
    )

    print(
        "Possession V5.1:",
        possession_path,
    )

    print(
        "Controller Gate:",
        gate_path,
    )

    print(
        "Air events:",
        events_path,
    )

    print(
        "V5.2 intermediate:",
        intermediate_path,
    )

    print(
        "V5.3 final:",
        output_path,
    )

    possession = pd.read_csv(
        possession_path
    )

    gate = pd.read_csv(
        gate_path
    )

    events = pd.read_csv(
        events_path
    )

    # ========================================================
    # PASS A
    # ========================================================

    print("")
    print(
        "=" * 62
    )
    print(
        "PASS A — POSSESSION V5.2 / "
        "CONTROLLER GATE INTEGRATION"
    )
    print(
        "=" * 62
    )

    (
        v5_2,
        v5_2_audit,
        v5_2_summary,
        fps_v5_2,

    ) = run_v5_2(
        possession,
        gate,
    )

    v5_2.to_csv(
        intermediate_path,
        index=False,
    )

    v5_2_summary.to_csv(
        intermediate_summary_path,
        index=False,
    )

    v5_2_audit.to_csv(
        intermediate_audit_path,
        index=False,
    )

    print(
        "Frames:",
        len(v5_2),
    )

    print(
        "FPS:",
        f"{fps_v5_2:.2f}",
    )

    print(
        "Overrides:",
        len(v5_2_audit),
    )

    print("")
    print(
        "V5.2 POSSESSION"
    )

    print(
        v5_2[
            "possession_v5_2"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print(
        "V5.2 BALL STATE"
    )

    print(
        v5_2[
            "ball_state_v5_2"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print(
        "V5.2 REGRESSION CHECKS"
    )

    v5_2_checks = (
        run_v5_2_checks(
            v5_2
        )
    )

    for (
        name,
        passed,

    ) in v5_2_checks.items():

        print(
            f"{'PASS' if passed else 'FAIL'}"
            f" | {name}"
        )

    # ========================================================
    # IMPORTANT:
    #
    # Preserve old serialization boundary:
    #
    # V5.2 -> CSV -> V5.3
    # ========================================================

    v5_2_reloaded = pd.read_csv(
        intermediate_path
    )

    # ========================================================
    # PASS B
    # ========================================================

    print("")
    print(
        "=" * 62
    )
    print(
        "PASS B — POSSESSION V5.3 / "
        "CONTEST EPISODE PERSISTENCE"
    )
    print(
        "=" * 62
    )

    (
        v5_3,
        v5_3_audit,
        v5_3_summary,
        contest_episodes,
        fps_v5_3,

    ) = run_v5_3(
        v5_2_reloaded,
        events,
    )

    v5_3.to_csv(
        output_path,
        index=False,
    )

    v5_3_summary.to_csv(
        summary_path,
        index=False,
    )

    v5_3_audit.to_csv(
        audit_path,
        index=False,
    )

    contest_episodes.to_csv(
        episodes_path,
        index=False,
    )

    print(
        "Frames:",
        len(v5_3),
    )

    print(
        "FPS:",
        f"{fps_v5_3:.2f}",
    )

    print(
        "Contest episodes:",
        len(
            contest_episodes
        ),
    )

    print(
        "Contest episode frames:",
        int(
            v5_3[
                "contest_episode_active"
            ]
            .sum()
        ),
    )

    print(
        "Frames changed from V5.2:",
        int(
            v5_3[
                "contest_changed_from_v5_2"
            ]
            .sum()
        ),
    )

    print("")
    print(
        "V5.3 POSSESSION"
    )

    print(
        v5_3[
            "possession_v5_3"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print(
        "V5.3 BALL STATE"
    )

    print(
        v5_3[
            "ball_state_v5_3"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print(
        "V5.3 REGRESSION CHECKS"
    )

    v5_3_checks = (
        run_v5_3_checks(
            v5_3
        )
    )

    for (
        name,
        passed,

    ) in v5_3_checks.items():

        print(
            f"{'PASS' if passed else 'FAIL'}"
            f" | {name}"
        )

    all_pass = (
        all(
            v5_2_checks.values()
        )

        and

        all(
            v5_3_checks.values()
        )
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
        "Built-in regressions:",
        (
            "PASS"
            if all_pass
            else
            "FAIL"
        ),
    )

    print(
        "V5.2 frames:",
        intermediate_path,
    )

    print(
        "V5.2 summary:",
        intermediate_summary_path,
    )

    print(
        "V5.2 audit:",
        intermediate_audit_path,
    )

    print(
        "V5.3 frames:",
        output_path,
    )

    print(
        "V5.3 summary:",
        summary_path,
    )

    print(
        "V5.3 audit:",
        audit_path,
    )

    print(
        "Contest episodes:",
        episodes_path,
    )

    if not all_pass:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
