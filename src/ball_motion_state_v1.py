import argparse
import os

import numpy as np
import pandas as pd


def clean_team(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value if value in {"A", "B"} else None


def clean_id(value):
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def clean_text(value, default=None):
    if pd.isna(value):
        return default
    value = str(value).strip()
    if value == "" or value.lower() in {"none", "nan"}:
        return default
    return value


def same_player(team_a, id_a, team_b, id_b):
    team_a = clean_team(team_a)
    team_b = clean_team(team_b)

    if (
        team_a is None
        or team_b is None
        or team_a != team_b
        or pd.isna(id_a)
        or pd.isna(id_b)
    ):
        return False

    return abs(float(id_a) - float(id_b)) < 0.1


def estimate_fps(df):
    if "time_sec" not in df.columns:
        return 25.0

    timing = (
        df[["frame", "time_sec"]]
        .dropna()
        .sort_values("frame")
    )

    if len(timing) < 2:
        return 25.0

    df_frame = timing["frame"].diff()
    df_time = timing["time_sec"].diff()

    valid = (
        df_frame.notna()
        & df_time.notna()
        & (df_frame > 0)
        & (df_time > 0)
    )

    fps_values = (
        df_frame[valid] / df_time[valid]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    return float(fps_values.median()) if len(fps_values) else 25.0


def group_keep_runs(gate_keep):
    runs = []
    current = []

    prev_frame = None
    prev_team = None
    prev_id = np.nan

    for _, row in gate_keep.sort_values("frame").iterrows():
        frame = int(row["frame"])
        team = clean_team(row["controller_team"])
        track_id = clean_id(row["controller_id"])

        if not current:
            current = [row.to_dict()]
        else:
            consecutive = frame == prev_frame + 1
            same = same_player(team, track_id, prev_team, prev_id)

            if consecutive and same:
                current.append(row.to_dict())
            else:
                runs.append(pd.DataFrame(current))
                current = [row.to_dict()]

        prev_frame = frame
        prev_team = team
        prev_id = track_id

    if current:
        runs.append(pd.DataFrame(current))

    return runs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ball Motion State V1. "
            "Separates control/transit state from aerial evidence."
        )
    )

    parser.add_argument(
        "--possession",
        default="outputs/possession_v5_3_frames.csv",
    )

    parser.add_argument(
        "--gate",
        default="outputs/controller_gate_v2.csv",
    )

    parser.add_argument(
        "--events",
        default="outputs/air_touch_v4_events.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs",
    )

    parser.add_argument(
        "--launch-min-speed",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--reception-confirm-sec",
        type=float,
        default=0.12,
    )

    parser.add_argument(
        "--dribble-return-sec",
        type=float,
        default=0.32,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frames_out = os.path.join(
        args.output_dir,
        "ball_motion_state_v1_frames.csv",
    )

    episodes_out = os.path.join(
        args.output_dir,
        "ball_motion_state_v1_episodes.csv",
    )

    receptions_out = os.path.join(
        args.output_dir,
        "ball_motion_state_v1_receptions.csv",
    )

    dribble_out = os.path.join(
        args.output_dir,
        "ball_motion_state_v1_dribble_gaps.csv",
    )

    # ============================================================
    # LOAD
    # ============================================================

    print("Loading Possession V5.3...")
    possession = pd.read_csv(args.possession)

    print("Loading Controller Gate V2...")
    gate = pd.read_csv(args.gate)

    print("Loading AirTouch V4 events...")
    events = pd.read_csv(args.events)

    for col in [
        "frame",
        "time_sec",
        "ball_speed_mps",
        "controller_id_v5_3",
        "last_touch_id_v5_3",
    ]:
        if col in possession.columns:
            possession[col] = pd.to_numeric(
                possession[col],
                errors="coerce",
            )

    possession = (
        possession
        .dropna(subset=["frame"])
        .copy()
        .sort_values("frame")
        .reset_index(drop=True)
    )

    possession["frame"] = possession["frame"].astype(int)

    possession_by_frame = possession.set_index("frame")

    frame_max = int(possession["frame"].max())

    fps = estimate_fps(possession)

    reception_confirm_frames = max(
        2,
        int(round(args.reception_confirm_sec * fps)),
    )

    dribble_return_frames = max(
        2,
        int(round(args.dribble_return_sec * fps)),
    )

    print(f"Estimated FPS: {fps:.2f}")
    print(
        f"Trusted reception: "
        f"{reception_confirm_frames} KEEP frames"
    )
    print(
        f"Dribble return window: "
        f"{dribble_return_frames} frames"
    )

    # ============================================================
    # CONTROLLER GATE
    # ============================================================

    numeric_gate_cols = [
        "frame",
        "controller_id",
        "ball_speed_mps",
        "foot_distance_px",
        "foot_distance_norm",
        "ball_rel_y",
    ]

    for col in numeric_gate_cols:
        if col in gate.columns:
            gate[col] = pd.to_numeric(
                gate[col],
                errors="coerce",
            )

    gate = gate.dropna(subset=["frame"]).copy()
    gate["frame"] = gate["frame"].astype(int)

    gate_keep = gate[
        (gate["gate_status"] == "KEEP")
        & gate["controller_team"].isin(["A", "B"])
        & gate["controller_id"].notna()
    ].copy()

    keep_runs = group_keep_runs(gate_keep)

    # ============================================================
    # TRUSTED RECEPTIONS
    # ============================================================

    reception_rows = []
    trusted_receptions = []

    for run_id, run in enumerate(keep_runs, start=1):
        run = (
            run
            .sort_values("frame")
            .reset_index(drop=True)
        )

        start = int(run.iloc[0]["frame"])
        end = int(run.iloc[-1]["frame"])

        team = clean_team(
            run.iloc[0]["controller_team"]
        )

        track_id = clean_id(
            run.iloc[0]["controller_id"]
        )

        run_length = len(run)

        trusted = (
            run_length
            >=
            reception_confirm_frames
        )

        record = {
            "run_id": run_id,
            "start_frame": start,
            "end_frame": end,
            "start_time_sec": start / fps,
            "end_time_sec": end / fps,
            "team": team,
            "track_id": track_id,
            "run_length": run_length,
            "trusted_reception": trusted,
        }

        reception_rows.append(record)

        if trusted:
            trusted_receptions.append(record)

    receptions_df = pd.DataFrame(reception_rows)
    receptions_df.to_csv(receptions_out, index=False)

    trusted_receptions = sorted(
        trusted_receptions,
        key=lambda r: r["start_frame"],
    )

    reception_start_map = {
        int(r["start_frame"]): r
        for r in trusted_receptions
    }

    print(
        f"Controller KEEP runs: "
        f"{len(keep_runs)}"
    )

    print(
        f"Trusted reception runs: "
        f"{len(trusted_receptions)}"
    )

    # ============================================================
    # PHYSICAL EVENT LAYER
    #
    # IMPORTANT:
    # We only trust physical event output here.
    # We DO NOT use possession_v5_3 AerialContest to create
    # aerial evidence. This removes the circular dependency.
    # ============================================================

    for col in [
        "physical_event_id",
        "start_frame",
        "end_frame",
        "last_touch_track_id",
    ]:
        if col in events.columns:
            events[col] = pd.to_numeric(
                events[col],
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

    contest_frames = set()
    air_touch_frames = set()
    weak_interaction_frames = set()

    air_touch_info = {}

    for _, event in events.iterrows():
        start = int(event["start_frame"])
        end = int(event["end_frame"])

        final_class = clean_text(
            event["final_class"],
            "",
        )

        if final_class == "AerialContest":
            contest_frames.update(
                range(start, end + 1)
            )

        elif final_class == "AirTouch":
            for frame in range(
                start,
                end + 1,
            ):
                air_touch_frames.add(frame)

                air_touch_info[frame] = {
                    "team": clean_team(
                        event.get(
                            "last_touch_team"
                        )
                    ),
                    "track_id": clean_id(
                        event.get(
                            "last_touch_track_id"
                        )
                    ),
                }

        elif final_class == "WeakInteraction":
            weak_interaction_frames.update(
                range(start, end + 1)
            )

        # ReceptionOrBounce is intentionally NOT used as a
        # ground-contact signal in V1 because the current
        # classifier does not reliably separate reception,
        # bounce, and close aerial interaction.

    # ============================================================
    # CONTROL-EXIT CANDIDATES
    # ============================================================

    launch_candidates = []

    for run in keep_runs:
        run = (
            run
            .sort_values("frame")
            .reset_index(drop=True)
        )

        run_end = int(
            run.iloc[-1]["frame"]
        )

        candidate_frame = run_end + 1

        if candidate_frame > frame_max:
            continue

        if (
            candidate_frame
            not in
            possession_by_frame.index
        ):
            continue

        current = possession_by_frame.loc[
            candidate_frame
        ]

        if isinstance(
            current,
            pd.DataFrame,
        ):
            current = current.iloc[0]

        state = clean_text(
            current.get(
                "ball_state_v5_3"
            ),
            "",
        )

        speed = current.get(
            "ball_speed_mps",
            np.nan,
        )

        if state not in {
            "InTransit",
            "Loose",
            "Uncontrolled",
        }:
            continue

        if (
            pd.isna(speed)
            or
            float(speed)
            <
            args.launch_min_speed
        ):
            continue

        launch_candidates.append(
            {
                "candidate_frame":
                    candidate_frame,

                "launch_team":
                    clean_team(
                        run.iloc[-1][
                            "controller_team"
                        ]
                    ),

                "launch_track_id":
                    clean_id(
                        run.iloc[-1][
                            "controller_id"
                        ]
                    ),

                "ball_speed_mps":
                    float(speed),
            }
        )

    # ============================================================
    # DRIBBLE GAP FILTER
    # ============================================================

    dribble_gaps = []
    dribble_frame_map = {}
    committed_launch_map = {}

    for candidate in launch_candidates:
        start = int(
            candidate[
                "candidate_frame"
            ]
        )

        launch_team = candidate[
            "launch_team"
        ]

        launch_id = candidate[
            "launch_track_id"
        ]

        next_reception = next(
            (
                r
                for r in trusted_receptions
                if int(
                    r["start_frame"]
                ) > start
            ),
            None,
        )

        is_dribble = False

        if next_reception is not None:
            reception_frame = int(
                next_reception[
                    "start_frame"
                ]
            )

            gap_frames = (
                reception_frame
                -
                start
            )

            same_return = same_player(
                launch_team,
                launch_id,
                next_reception["team"],
                next_reception["track_id"],
            )

            has_physical_air_event = any(
                (
                    frame in contest_frames
                    or
                    frame in air_touch_frames
                )
                for frame in range(
                    start,
                    reception_frame,
                )
            )

            if (
                same_return
                and
                gap_frames
                <=
                dribble_return_frames
                and
                not has_physical_air_event
            ):
                is_dribble = True

                gap = {
                    "start_frame": start,
                    "end_frame":
                        reception_frame - 1,
                    "duration_frames":
                        gap_frames,
                    "team":
                        launch_team,
                    "track_id":
                        launch_id,
                    "reception_frame":
                        reception_frame,
                }

                dribble_gaps.append(gap)

                for frame in range(
                    start,
                    reception_frame,
                ):
                    dribble_frame_map[
                        frame
                    ] = gap

        if not is_dribble:
            committed_launch_map[
                start
            ] = candidate

    dribble_df = pd.DataFrame(
        dribble_gaps
    )

    dribble_df.to_csv(
        dribble_out,
        index=False,
    )

    # ============================================================
    # MOTION STATE MACHINE
    #
    # Key design:
    #
    # Control state:
    #   Controlled / TransitUncontrolled
    #
    # Flight evidence:
    #   None / AerialContest / AirTouch
    #
    # We deliberately do NOT infer continuous airborne state
    # without continuous evidence.
    # ============================================================

    output_rows = []
    episode_rows = []

    transit_active = False
    episode_id = 0

    episode_start = None
    episode_launch_reason = None
    episode_launch_team = None
    episode_launch_id = np.nan

    episode_had_contest = False
    episode_had_air_touch = False
    episode_had_weak_interaction = False

    episode_last_touch_team = None
    episode_last_touch_id = np.nan

    def close_episode(
        end_frame,
        resolution_type,
        receiver_team=None,
        receiver_id=np.nan,
    ):
        nonlocal transit_active
        nonlocal episode_start
        nonlocal episode_launch_reason
        nonlocal episode_launch_team
        nonlocal episode_launch_id
        nonlocal episode_had_contest
        nonlocal episode_had_air_touch
        nonlocal episode_had_weak_interaction
        nonlocal episode_last_touch_team
        nonlocal episode_last_touch_id

        if (
            episode_start is None
            or
            end_frame < episode_start
        ):
            transit_active = False
            return

        same_team_pass = (
            resolution_type
            ==
            "trusted_reception"

            and

            episode_launch_reason
            ==
            "trusted_control_exit"

            and

            episode_launch_team
            is not None

            and

            receiver_team
            is not None

            and

            episode_launch_team
            ==
            receiver_team

            and

            pd.notna(
                episode_launch_id
            )

            and

            pd.notna(
                receiver_id
            )

            and

            not same_player(
                episode_launch_team,
                episode_launch_id,
                receiver_team,
                receiver_id,
            )

            and

            not episode_had_contest

            and

            not episode_had_air_touch
        )

        turnover_candidate = (
            resolution_type
            ==
            "trusted_reception"

            and

            episode_launch_team
            is not None

            and

            receiver_team
            is not None

            and

            episode_launch_team
            !=
            receiver_team
        )

        if same_team_pass:
            event_class = (
                "ConfirmedPass"
            )

        elif turnover_candidate:
            event_class = (
                "TurnoverCandidate"
            )

        elif (
            resolution_type
            ==
            "trusted_reception"
        ):
            event_class = (
                "ResolvedTransit"
            )

        else:
            event_class = (
                "UnresolvedTransit"
            )

        episode_rows.append(
            {
                "episode_id":
                    episode_id,

                "start_frame":
                    episode_start,

                "end_frame":
                    end_frame,

                "duration_frames":
                    (
                        end_frame
                        -
                        episode_start
                        +
                        1
                    ),

                "duration_sec":
                    (
                        end_frame
                        -
                        episode_start
                        +
                        1
                    )
                    /
                    fps,

                "launch_reason":
                    episode_launch_reason,

                "launch_team":
                    episode_launch_team,

                "launch_track_id":
                    episode_launch_id,

                "had_aerial_contest":
                    episode_had_contest,

                "had_air_touch":
                    episode_had_air_touch,

                "had_weak_interaction":
                    episode_had_weak_interaction,

                "last_touch_team":
                    episode_last_touch_team,

                "last_touch_track_id":
                    episode_last_touch_id,

                "resolution_type":
                    resolution_type,

                "receiver_team":
                    receiver_team,

                "receiver_track_id":
                    receiver_id,

                "event_class":
                    event_class,
            }
        )

        transit_active = False
        episode_start = None
        episode_launch_reason = None
        episode_launch_team = None
        episode_launch_id = np.nan
        episode_had_contest = False
        episode_had_air_touch = False
        episode_had_weak_interaction = False
        episode_last_touch_team = None
        episode_last_touch_id = np.nan

    # ============================================================
    # FRAME LOOP
    # ============================================================

    for _, base in possession.iterrows():
        frame = int(base["frame"])

        baseline_state = clean_text(
            base.get(
                "ball_state_v5_3"
            ),
            "Unknown",
        )

        baseline_possession = clean_text(
            base.get(
                "possession_v5_3"
            ),
            "Unknown",
        )

        baseline_controller_team = clean_team(
            base.get(
                "controller_team_v5_3"
            )
        )

        baseline_controller_id = clean_id(
            base.get(
                "controller_id_v5_3"
            )
        )

        reception_here = (
            reception_start_map.get(
                frame
            )
        )

        # --------------------------------------------------------
        # TRUSTED RECEPTION ENDS TRANSIT
        # --------------------------------------------------------

        if (
            transit_active
            and
            reception_here
            is not None
            and
            frame > episode_start
        ):
            close_episode(
                end_frame=frame - 1,
                resolution_type=(
                    "trusted_reception"
                ),
                receiver_team=(
                    reception_here[
                        "team"
                    ]
                ),
                receiver_id=(
                    reception_here[
                        "track_id"
                    ]
                ),
            )

        # --------------------------------------------------------
        # START TRANSIT
        #
        # A control exit starts generic transit.
        # A physical aerial event can also create an
        # uncontrolled transit episode when no launch owner is
        # known.
        # --------------------------------------------------------

        if not transit_active:
            launch_reason = None
            launch_team = None
            launch_id = np.nan

            if frame in committed_launch_map:
                info = committed_launch_map[
                    frame
                ]

                launch_reason = (
                    "trusted_control_exit"
                )

                launch_team = info[
                    "launch_team"
                ]

                launch_id = info[
                    "launch_track_id"
                ]

            elif frame in contest_frames:
                launch_reason = (
                    "physical_aerial_contest"
                )

            elif frame in air_touch_frames:
                launch_reason = (
                    "physical_air_touch"
                )

                info = air_touch_info.get(
                    frame,
                    {},
                )

                launch_team = clean_team(
                    info.get("team")
                )

                launch_id = clean_id(
                    info.get("track_id")
                )

            if launch_reason is not None:
                transit_active = True
                episode_id += 1

                episode_start = frame
                episode_launch_reason = (
                    launch_reason
                )

                episode_launch_team = (
                    launch_team
                )

                episode_launch_id = (
                    launch_id
                )

                episode_last_touch_team = (
                    launch_team
                )

                episode_last_touch_id = (
                    launch_id
                )

        row = base.to_dict()

        row[
            "motion_episode_id_v1"
        ] = (
            episode_id
            if transit_active
            else
            np.nan
        )

        row[
            "motion_active_v1"
        ] = transit_active

        row[
            "controller_suppressed_by_motion_v1"
        ] = False

        row[
            "dribble_gap_v1"
        ] = (
            frame
            in
            dribble_frame_map
        )

        # --------------------------------------------------------
        # DRIBBLE GAP
        # --------------------------------------------------------

        if (
            not transit_active
            and
            frame in dribble_frame_map
        ):
            gap = dribble_frame_map[
                frame
            ]

            row[
                "motion_state_v1"
            ] = "DribbleGap"

            row[
                "flight_evidence_v1"
            ] = "None"

            row[
                "possession_motion_v1"
            ] = gap["team"]

            row[
                "controller_team_motion_v1"
            ] = gap["team"]

            row[
                "controller_id_motion_v1"
            ] = gap["track_id"]

            row[
                "last_touch_team_motion_v1"
            ] = gap["team"]

            row[
                "last_touch_id_motion_v1"
            ] = gap["track_id"]

        # --------------------------------------------------------
        # ACTIVE TRANSIT
        # --------------------------------------------------------

        elif transit_active:
            row[
                "controller_team_motion_v1"
            ] = None

            row[
                "controller_id_motion_v1"
            ] = np.nan

            if (
                baseline_controller_team
                is not None
                and
                pd.notna(
                    baseline_controller_id
                )
            ):
                row[
                    "controller_suppressed_by_motion_v1"
                ] = True

            if frame in contest_frames:
                episode_had_contest = True

                episode_last_touch_team = None
                episode_last_touch_id = np.nan

                row[
                    "motion_state_v1"
                ] = "AerialContest"

                row[
                    "flight_evidence_v1"
                ] = "AerialContest"

                row[
                    "possession_motion_v1"
                ] = "Contested"

            elif frame in air_touch_frames:
                episode_had_air_touch = True

                info = air_touch_info.get(
                    frame,
                    {},
                )

                episode_last_touch_team = clean_team(
                    info.get("team")
                )

                episode_last_touch_id = clean_id(
                    info.get("track_id")
                )

                row[
                    "motion_state_v1"
                ] = "AirTouchInTransit"

                row[
                    "flight_evidence_v1"
                ] = "AirTouch"

                row[
                    "possession_motion_v1"
                ] = "Unknown"

            elif frame in weak_interaction_frames:
                episode_had_weak_interaction = True

                row[
                    "motion_state_v1"
                ] = "TransitUncontrolled"

                row[
                    "flight_evidence_v1"
                ] = "WeakInteraction"

                row[
                    "possession_motion_v1"
                ] = "Unknown"

            elif baseline_state == "Missing":
                row[
                    "motion_state_v1"
                ] = "TransitMissing"

                row[
                    "flight_evidence_v1"
                ] = "None"

                row[
                    "possession_motion_v1"
                ] = "Unknown"

            else:
                row[
                    "motion_state_v1"
                ] = "TransitUncontrolled"

                row[
                    "flight_evidence_v1"
                ] = "None"

                row[
                    "possession_motion_v1"
                ] = "Unknown"

            row[
                "last_touch_team_motion_v1"
            ] = (
                episode_last_touch_team
            )

            row[
                "last_touch_id_motion_v1"
            ] = (
                episode_last_touch_id
            )

        # --------------------------------------------------------
        # RESOLVED / CONTROLLED / BASELINE
        # --------------------------------------------------------

        else:
            row[
                "motion_state_v1"
            ] = "GroundOrControlled"

            row[
                "flight_evidence_v1"
            ] = "None"

            row[
                "possession_motion_v1"
            ] = baseline_possession

            row[
                "controller_team_motion_v1"
            ] = baseline_controller_team

            row[
                "controller_id_motion_v1"
            ] = baseline_controller_id

            row[
                "last_touch_team_motion_v1"
            ] = clean_team(
                base.get(
                    "last_touch_team_v5_3"
                )
            )

            row[
                "last_touch_id_motion_v1"
            ] = clean_id(
                base.get(
                    "last_touch_id_v5_3"
                )
            )

        output_rows.append(row)

    if transit_active:
        close_episode(
            end_frame=frame_max,
            resolution_type="video_end",
        )

    frames_df = pd.DataFrame(
        output_rows
    )

    episodes_df = pd.DataFrame(
        episode_rows
    )

    # ============================================================
    # RETROSPECTIVE EVENT LABELS
    # ============================================================

    if len(episodes_df) > 0:
        for _, episode in episodes_df.iterrows():
            eid = int(
                episode["episode_id"]
            )

            event_class = clean_text(
                episode["event_class"],
                "",
            )

            launch_team = clean_team(
                episode["launch_team"]
            )

            mask = (
                frames_df[
                    "motion_episode_id_v1"
                ]
                ==
                eid
            )

            if event_class == "ConfirmedPass":
                generic_mask = (
                    mask
                    &
                    frames_df[
                        "motion_state_v1"
                    ].isin(
                        [
                            "TransitUncontrolled",
                            "TransitMissing",
                        ]
                    )
                )

                frames_df.loc[
                    generic_mask,
                    "motion_state_v1",
                ] = "ConfirmedPassInTransit"

                frames_df.loc[
                    generic_mask,
                    "possession_motion_v1",
                ] = launch_team

    # ============================================================
    # SAVE
    # ============================================================

    frames_df.to_csv(
        frames_out,
        index=False,
    )

    episodes_df.to_csv(
        episodes_out,
        index=False,
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    print("")
    print("=" * 52)
    print("BALL MOTION STATE V1 SUMMARY")
    print("=" * 52)

    print(
        f"Frames: {len(frames_df)}"
    )

    print(
        "Motion-active frames:",
        int(
            frames_df[
                "motion_active_v1"
            ].sum()
        ),
    )

    print(
        "DribbleGap frames:",
        int(
            frames_df[
                "dribble_gap_v1"
            ].sum()
        ),
    )

    print(
        "Motion episodes:",
        len(episodes_df),
    )

    print(
        "Controllers suppressed:",
        int(
            frames_df[
                "controller_suppressed_by_motion_v1"
            ].sum()
        ),
    )

    print("")
    print("MOTION STATES")

    print(
        frames_df[
            "motion_state_v1"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print("FLIGHT EVIDENCE")

    print(
        frames_df[
            "flight_evidence_v1"
        ]
        .value_counts()
        .to_string()
    )

    print("")
    print("=" * 52)
    print("DRIBBLE GAPS")
    print("=" * 52)

    if len(dribble_df):
        print(
            dribble_df.to_string(
                index=False
            )
        )
    else:
        print("None")

    print("")
    print("=" * 52)
    print("MOTION EPISODES")
    print("=" * 52)

    if len(episodes_df):
        display_cols = [
            "episode_id",
            "start_frame",
            "end_frame",
            "duration_sec",
            "launch_reason",
            "launch_team",
            "launch_track_id",
            "had_aerial_contest",
            "had_air_touch",
            "resolution_type",
            "receiver_team",
            "receiver_track_id",
            "event_class",
        ]

        print(
            episodes_df[
                display_cols
            ]
            .round(3)
            .to_string(
                index=False
            )
        )

    print("")
    print("Outputs:")
    print("Frames:", frames_out)
    print("Episodes:", episodes_out)
    print("Receptions:", receptions_out)
    print("Dribble gaps:", dribble_out)


if __name__ == "__main__":
    main()
