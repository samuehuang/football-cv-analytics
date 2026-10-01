#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MAX_AIR_TOUCH_CARRY_FRAMES = 50
MAX_AIR_TOUCH_MISSING_CARRY_FRAMES = 8

FRAME_159 = 159
FRAME_172 = 172
FRAME_203 = 203
FRAME_253 = 253
FRAME_281 = 281
FRAME_287 = 287


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

    return (
        value
        if value in {"A", "B"}
        else None
    )


def clean_id(value):
    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except Exception:
        return np.nan


def clean_source(value):
    if pd.isna(value):
        return "baseline_v4"

    text = str(value).strip()

    if (
        not text
        or
        text.lower()
        in {"none", "nan"}
    ):
        return "baseline_v4"

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
    if "time_sec" not in df.columns:
        return 25.0

    timing = (
        df[
            [
                "frame",
                "time_sec",
            ]
        ]
        .dropna()
        .sort_values("frame")
    )

    if len(timing) < 2:
        return 25.0

    frame_diff = (
        timing["frame"]
        .diff()
    )

    time_diff = (
        timing["time_sec"]
        .diff()
    )

    valid = (
        frame_diff.notna()
        &
        time_diff.notna()
        &
        (frame_diff > 0)
        &
        (time_diff > 0)
    )

    if not valid.any():
        return 25.0

    values = (
        frame_diff[valid]
        /
        time_diff[valid]
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return 25.0

    return float(
        values.median()
    )


def run_v5_1(
    v4,
    events,
):

    # ========================================================
    # Normalize V4
    # ========================================================

    v4 = v4.copy()

    v4["frame"] = pd.to_numeric(
        v4["frame"],
        errors="coerce",
    )

    v4 = (
        v4
        .dropna(
            subset=["frame"]
        )
        .copy()
    )

    v4["frame"] = (
        v4["frame"]
        .astype(int)
    )

    for column in [
        "time_sec",
        "controller_id_v4",
    ]:

        if column in v4.columns:

            v4[column] = pd.to_numeric(
                v4[column],
                errors="coerce",
            )

    v4 = (
        v4
        .sort_values("frame")
        .reset_index(drop=True)
    )

    fps = estimate_fps(
        v4
    )

    # ========================================================
    # Normalize Air Interaction events
    # ========================================================

    events = events.copy()

    for column in [
        "physical_event_id",
        "start_frame",
        "end_frame",
        "last_touch_track_id",
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

    # ========================================================
    # Strong physical events only
    # ========================================================

    strong_events = (
        events[
            events[
                "final_class"
            ].isin(
                [
                    "AirTouch",
                    "AerialContest",
                ]
            )
        ]
        .copy()
    )

    event_by_frame = {}

    for _, event in (
        strong_events.iterrows()
    ):

        start_frame = int(
            event["start_frame"]
        )

        end_frame = int(
            event["end_frame"]
        )

        for frame in range(
            start_frame,
            end_frame + 1,
        ):

            if (
                frame
                not in
                event_by_frame
            ):

                event_by_frame[
                    frame
                ] = event

            else:

                # AerialContest wins overlaps.
                if (
                    str(
                        event[
                            "final_class"
                        ]
                    )
                    ==
                    "AerialContest"
                ):

                    event_by_frame[
                        frame
                    ] = event

    # ========================================================
    # State memory
    # ========================================================

    air_owner_team = None
    air_owner_id = np.nan
    air_owner_start_frame = None

    last_touch_team = None
    last_touch_id = np.nan
    last_touch_type = None

    rows = []
    audit_rows = []

    # ========================================================
    # Main loop
    # ========================================================

    for _, base in (
        v4.iterrows()
    ):

        frame = int(
            base["frame"]
        )

        row = (
            base.to_dict()
        )

        base_possession = str(
            base.get(
                "possession_v4",
                "Unknown",
            )
        )

        base_ball_state = str(
            base.get(
                "ball_state_v4",
                "Unknown",
            )
        )

        base_controller_team = (
            clean_team(
                base.get(
                    "controller_team_v4",
                    None,
                )
            )
        )

        base_controller_id = (
            clean_id(
                base.get(
                    "controller_id_v4",
                    np.nan,
                )
            )
        )

        # Preserve legacy V5.1 exactly.
        #
        # Original code asks for "source_v4".
        # It does NOT read "possession_source_v4".
        base_source = (
            clean_source(
                base.get(
                    "source_v4",
                    None,
                )
            )
        )

        has_controller = (
            base_controller_team
            in {"A", "B"}

            and

            pd.notna(
                base_controller_id
            )
        )

        # ====================================================
        # Default V5 = V4
        # ====================================================

        possession_v5 = (
            base_possession
        )

        ball_state_v5 = (
            base_ball_state
        )

        controller_team_v5 = (
            base_controller_team
        )

        controller_id_v5 = (
            base_controller_id
        )

        source_v5 = (
            base_source
        )

        physical_event_class = None
        physical_event_id = np.nan
        physical_participants = None

        event = event_by_frame.get(
            frame,
            None,
        )

        # ====================================================
        # PRIORITY 1
        # Physical AerialContest
        # ====================================================

        if (
            event is not None

            and

            str(
                event[
                    "final_class"
                ]
            )
            ==
            "AerialContest"
        ):

            physical_event_class = (
                "AerialContest"
            )

            physical_event_id = (
                event.get(
                    "physical_event_id",
                    np.nan,
                )
            )

            physical_participants = (
                event.get(
                    "participants",
                    None,
                )
            )

            possession_v5 = (
                "Contested"
            )

            ball_state_v5 = (
                "AerialContest"
            )

            controller_team_v5 = None
            controller_id_v5 = np.nan

            source_v5 = (
                "physical_aerial_contest"
            )

            air_owner_team = None
            air_owner_id = np.nan
            air_owner_start_frame = None

            last_touch_team = None
            last_touch_id = np.nan

            last_touch_type = (
                "AerialContestUnknown"
            )

        # ====================================================
        # PRIORITY 2
        # Confirmed physical AirTouch
        # ====================================================

        elif (
            event is not None

            and

            str(
                event[
                    "final_class"
                ]
            )
            ==
            "AirTouch"
        ):

            touch_team = (
                clean_team(
                    event.get(
                        "last_touch_team",
                        None,
                    )
                )
            )

            touch_id = (
                clean_id(
                    event.get(
                        "last_touch_track_id",
                        np.nan,
                    )
                )
            )

            physical_event_class = (
                "AirTouch"
            )

            physical_event_id = (
                event.get(
                    "physical_event_id",
                    np.nan,
                )
            )

            physical_participants = (
                event.get(
                    "participants",
                    None,
                )
            )

            if (
                touch_team
                in {"A", "B"}
            ):

                possession_v5 = (
                    touch_team
                )

                ball_state_v5 = (
                    "InTransit"
                )

                controller_team_v5 = None
                controller_id_v5 = np.nan

                source_v5 = (
                    "confirmed_air_touch"
                )

                air_owner_team = (
                    touch_team
                )

                air_owner_id = (
                    touch_id
                )

                air_owner_start_frame = (
                    frame
                )

                last_touch_team = (
                    touch_team
                )

                last_touch_id = (
                    touch_id
                )

                last_touch_type = (
                    "AirTouch"
                )

        # ====================================================
        # PRIORITY 3
        # Confirmed V4 controller
        # ====================================================

        elif has_controller:

            possession_v5 = (
                base_controller_team
            )

            ball_state_v5 = (
                base_ball_state
            )

            controller_team_v5 = (
                base_controller_team
            )

            controller_id_v5 = (
                base_controller_id
            )

            source_v5 = (
                base_source
            )

            air_owner_team = None
            air_owner_id = np.nan
            air_owner_start_frame = None

            last_touch_team = (
                base_controller_team
            )

            last_touch_id = (
                base_controller_id
            )

            last_touch_type = (
                "Controlled"
            )

        # ====================================================
        # PRIORITY 4
        # V4 contest
        # ====================================================

        elif (
            base_ball_state
            in {
                "AerialContest",
                "Contested",
            }

            or

            base_possession
            ==
            "Contested"
        ):

            possession_v5 = (
                base_possession
            )

            ball_state_v5 = (
                base_ball_state
            )

            controller_team_v5 = None
            controller_id_v5 = np.nan

            source_v5 = (
                base_source
            )

            air_owner_team = None
            air_owner_id = np.nan
            air_owner_start_frame = None

            last_touch_team = None
            last_touch_id = np.nan

            last_touch_type = (
                "ContestUnknown"
            )

        # ====================================================
        # PRIORITY 5
        # Carry ownership after AirTouch
        # ====================================================

        elif (
            air_owner_team
            in {"A", "B"}

            and

            air_owner_start_frame
            is not None
        ):

            carry_age = (
                frame
                -
                air_owner_start_frame
            )

            carry_allowed = False

            if (
                base_ball_state
                in {
                    "InTransit",
                    "Loose",
                }
            ):

                carry_allowed = (
                    carry_age
                    <=
                    MAX_AIR_TOUCH_CARRY_FRAMES
                )

            elif (
                base_ball_state
                ==
                "Missing"
            ):

                carry_allowed = (
                    carry_age
                    <=
                    MAX_AIR_TOUCH_MISSING_CARRY_FRAMES
                )

            if carry_allowed:

                possession_v5 = (
                    air_owner_team
                )

                ball_state_v5 = (
                    base_ball_state
                )

                controller_team_v5 = None
                controller_id_v5 = np.nan

                source_v5 = (
                    "carried_after_air_touch"
                )

            else:

                air_owner_team = None
                air_owner_id = np.nan
                air_owner_start_frame = None

                possession_v5 = (
                    base_possession
                )

                ball_state_v5 = (
                    base_ball_state
                )

                controller_team_v5 = (
                    base_controller_team
                )

                controller_id_v5 = (
                    base_controller_id
                )

                source_v5 = (
                    base_source
                )

                if (
                    last_touch_type
                    ==
                    "AirTouch"
                ):

                    last_touch_team = None
                    last_touch_id = np.nan
                    last_touch_type = None

                if (
                    last_touch_type
                    in {
                        "ContestUnknown",
                        "AerialContestUnknown",
                    }
                ):

                    last_touch_team = None
                    last_touch_id = np.nan
                    last_touch_type = None

        # ====================================================
        # PRIORITY 6
        # Normal V4 baseline
        # ====================================================

        else:

            possession_v5 = (
                base_possession
            )

            ball_state_v5 = (
                base_ball_state
            )

            controller_team_v5 = (
                base_controller_team
            )

            controller_id_v5 = (
                base_controller_id
            )

            source_v5 = (
                base_source
            )

            if (
                last_touch_type
                in {
                    "ContestUnknown",
                    "AerialContestUnknown",
                }
            ):

                last_touch_team = None
                last_touch_id = np.nan
                last_touch_type = None

        # ====================================================
        # Save
        # ====================================================

        row[
            "possession_v5"
        ] = possession_v5

        row[
            "ball_state_v5"
        ] = ball_state_v5

        row[
            "controller_team_v5"
        ] = controller_team_v5

        row[
            "controller_id_v5"
        ] = controller_id_v5

        row[
            "last_touch_team_v5"
        ] = last_touch_team

        row[
            "last_touch_id_v5"
        ] = last_touch_id

        row[
            "last_touch_type_v5"
        ] = last_touch_type

        row[
            "source_v5"
        ] = source_v5

        row[
            "physical_event_class_v5"
        ] = physical_event_class

        row[
            "physical_event_id_v5"
        ] = physical_event_id

        row[
            "physical_event_participants_v5"
        ] = physical_participants

        rows.append(
            row
        )

        # ====================================================
        # Audit
        # ====================================================

        changed = (
            str(
                possession_v5
            )
            !=
            str(
                base_possession
            )

            or

            str(
                ball_state_v5
            )
            !=
            str(
                base_ball_state
            )

            or

            clean_team(
                controller_team_v5
            )
            !=
            clean_team(
                base_controller_team
            )

            or

            physical_event_class
            is not None

            or

            source_v5
            ==
            "carried_after_air_touch"
        )

        if changed:

            audit_rows.append(
                {
                    "frame":
                        frame,

                    "time_sec":
                        base.get(
                            "time_sec",
                            frame / fps,
                        ),

                    "possession_v4":
                        base_possession,

                    "possession_v5":
                        possession_v5,

                    "ball_state_v4":
                        base_ball_state,

                    "ball_state_v5":
                        ball_state_v5,

                    "controller_team_v4":
                        base_controller_team,

                    "controller_id_v4":
                        base_controller_id,

                    "controller_team_v5":
                        controller_team_v5,

                    "controller_id_v5":
                        controller_id_v5,

                    "last_touch_team_v5":
                        last_touch_team,

                    "last_touch_id_v5":
                        last_touch_id,

                    "last_touch_type_v5":
                        last_touch_type,

                    "source_v4":
                        base_source,

                    "source_v5":
                        source_v5,

                    "physical_event_class":
                        physical_event_class,

                    "physical_event_id":
                        physical_event_id,

                    "participants":
                        physical_participants,
                }
            )

    v5 = pd.DataFrame(
        rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_rows = []

    for key, value in (
        v5[
            "possession_v5"
        ]
        .value_counts()
        .items()
    ):

        summary_rows.append(
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

    for key, value in (
        v5[
            "ball_state_v5"
        ]
        .value_counts()
        .items()
    ):

        summary_rows.append(
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

    summary = pd.DataFrame(
        summary_rows
    )

    return (
        v5,
        audit,
        summary,
        fps,
    )


def get_frame(
    df,
    frame,
):

    rows = df[
        df["frame"]
        ==
        frame
    ]

    if len(rows) == 0:
        return None

    return rows.iloc[0]


def run_regressions(
    v5,
):

    r159 = get_frame(
        v5,
        FRAME_159,
    )

    r172 = get_frame(
        v5,
        FRAME_172,
    )

    r203 = get_frame(
        v5,
        FRAME_203,
    )

    r253 = get_frame(
        v5,
        FRAME_253,
    )

    r281 = get_frame(
        v5,
        FRAME_281,
    )

    r287 = get_frame(
        v5,
        FRAME_287,
    )

    reg159 = (
        r159 is not None
        and
        str(
            r159["possession_v5"]
        )
        ==
        "B"
        and
        str(
            r159["ball_state_v5"]
        )
        ==
        "InTransit"
        and
        clean_team(
            r159[
                "controller_team_v5"
            ]
        )
        is None
        and
        pd.isna(
            r159[
                "last_touch_id_v5"
            ]
        )
        and
        pd.isna(
            r159[
                "last_touch_team_v5"
            ]
        )
    )

    reg172 = (
        r172 is not None
        and
        str(
            r172[
                "possession_v5"
            ]
        )
        ==
        "Contested"
        and
        str(
            r172[
                "ball_state_v5"
            ]
        )
        ==
        "AerialContest"
        and
        clean_team(
            r172[
                "controller_team_v5"
            ]
        )
        is None
        and
        str(
            r172[
                "last_touch_type_v5"
            ]
        )
        ==
        "AerialContestUnknown"
    )

    reg203 = (
        r203 is not None
        and
        str(
            r203[
                "source_v5"
            ]
        )
        not in {
            "confirmed_air_touch",
            "carried_after_air_touch",
        }
        and
        str(
            r203[
                "possession_v5"
            ]
        )
        ==
        str(
            r203[
                "possession_v4"
            ]
        )
        and
        pd.isna(
            r203[
                "last_touch_team_v5"
            ]
        )
        and
        pd.isna(
            r203[
                "last_touch_id_v5"
            ]
        )
    )

    reg253 = (
        r253 is not None
        and
        str(
            r253[
                "possession_v5"
            ]
        )
        ==
        "Contested"
        and
        str(
            r253[
                "ball_state_v5"
            ]
        )
        ==
        "AerialContest"
        and
        clean_team(
            r253[
                "controller_team_v5"
            ]
        )
        is None
        and
        str(
            r253[
                "last_touch_type_v5"
            ]
        )
        ==
        "ContestUnknown"
    )

    reg281 = (
        r281 is not None
        and
        str(
            r281[
                "possession_v5"
            ]
        )
        ==
        "B"
        and
        str(
            r281[
                "ball_state_v5"
            ]
        )
        ==
        "InTransit"
        and
        clean_team(
            r281[
                "controller_team_v5"
            ]
        )
        is None
        and
        clean_team(
            r281[
                "last_touch_team_v5"
            ]
        )
        ==
        "B"
        and
        same_player_id(
            r281[
                "last_touch_id_v5"
            ],
            10,
        )
        and
        str(
            r281[
                "last_touch_type_v5"
            ]
        )
        ==
        "AirTouch"
    )

    reg287 = (
        r287 is not None
        and
        str(
            r287[
                "possession_v5"
            ]
        )
        ==
        str(
            r287[
                "possession_v4"
            ]
        )
        and
        str(
            r287[
                "source_v5"
            ]
        )
        !=
        "unresolved_after_aerial_contest"
        and
        pd.isna(
            r287[
                "last_touch_team_v5"
            ]
        )
        and
        pd.isna(
            r287[
                "last_touch_id_v5"
            ]
        )
    )

    return {
        "frame_159_long_ball":
            reg159,

        "frame_172_aerial_contest":
            reg172,

        "frame_203_reception_bounce":
            reg203,

        "frame_253_contest":
            reg253,

        "frame_281_air_touch":
            reg281,

        "frame_287_post_contest":
            reg287,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Possession V5.1 canonical CLI: "
            "AirTouch / AerialContest "
            "LastTouch integration."
        )
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
        "--summary-output",
        default=None,
    )

    parser.add_argument(
        "--audit-output",
        default=None,
    )

    args = parser.parse_args()

    possession_path = Path(
        args.possession
    )

    events_path = Path(
        args.events
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
        "possession_v5_1_summary.csv"
    )

    audit_path = (
        Path(
            args.audit_output
        )
        if args.audit_output
        else
        output_path.parent
        /
        "possession_v5_1_audit.csv"
    )

    require_file(
        possession_path,
        "Possession seed CSV",
    )

    require_file(
        events_path,
        "Air interaction event CSV",
    )

    ensure_parent(
        output_path
    )

    ensure_parent(
        summary_path
    )

    ensure_parent(
        audit_path
    )

    print("")
    print(
        "=" * 58
    )
    print(
        "POSSESSION V5.1 CLI"
    )
    print(
        "=" * 58
    )

    print(
        "Possession:",
        possession_path,
    )

    print(
        "Events:",
        events_path,
    )

    print(
        "Output:",
        output_path,
    )

    v4 = pd.read_csv(
        possession_path
    )

    events = pd.read_csv(
        events_path
    )

    required_v4 = [
        "frame",
        "possession_v4",
        "ball_state_v4",
        "controller_team_v4",
        "controller_id_v4",
    ]

    missing_v4 = [
        column

        for column
        in required_v4

        if column
        not in
        v4.columns
    ]

    if missing_v4:

        raise ValueError(
            "Possession CSV missing required columns: "
            +
            ", ".join(
                missing_v4
            )
        )

    required_events = [
        "start_frame",
        "end_frame",
        "final_class",
    ]

    missing_events = [
        column

        for column
        in required_events

        if column
        not in
        events.columns
    ]

    if missing_events:

        raise ValueError(
            "Event CSV missing required columns: "
            +
            ", ".join(
                missing_events
            )
        )

    (
        v5,
        audit,
        summary,
        fps,

    ) = run_v5_1(
        v4,
        events,
    )

    v5.to_csv(
        output_path,
        index=False,
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    checks = run_regressions(
        v5
    )

    print("")
    print(
        "=" * 58
    )
    print(
        "POSSESSION V5.1 SUMMARY"
    )
    print(
        "=" * 58
    )

    print(
        "Frames:",
        len(v5),
    )

    print(
        "FPS:",
        f"{fps:.2f}",
    )

    print("")
    print(
        "FINAL POSSESSION"
    )

    print(
        v5[
            "possession_v5"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print(
        "FINAL BALL STATE"
    )

    print(
        v5[
            "ball_state_v5"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print(
        "V5 SOURCES"
    )

    print(
        v5[
            "source_v5"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print(
        "Changed / audited frames:",
        len(audit),
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

    print(
        "Audit CSV:",
        audit_path,
    )


if __name__ == "__main__":
    main()
