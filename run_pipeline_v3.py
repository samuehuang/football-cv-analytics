#!/usr/bin/env python3

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time

from datetime import (
    datetime,
    timezone,
)

from pathlib import Path

import cv2
import yaml


# ============================================================
# Project Root
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


# ============================================================
# General Helpers
# ============================================================

def now_iso():

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def resolve_path(
    path,
):

    path = Path(
        path
    )

    if path.is_absolute():

        return path

    return (
        ROOT
        /
        path
    ).resolve()


def require_file(
    path,
    label,
):

    path = Path(
        path
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"{label} not found: "
            f"{path}"
        )

    if (
        path.stat().st_size
        ==
        0
    ):

        raise RuntimeError(
            f"{label} is empty: "
            f"{path}"
        )


def sha256_file(
    path,
):

    h = hashlib.sha256()

    with Path(
        path
    ).open(
        "rb"
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            h.update(
                chunk
            )

    return (
        h.hexdigest()
    )


def get_git_commit():

    try:

        return (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=ROOT,
                text=True,
                stderr=(
                    subprocess.DEVNULL
                ),
            )
            .strip()
        )

    except Exception:

        return None


def get_video_info(
    video_path,
):

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
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

    cap.release()

    if fps is None or fps <= 0:

        raise RuntimeError(
            "Invalid video FPS."
        )

    return {
        "fps":
            float(
                fps
            ),

        "frame_count":
            frame_count,

        "width":
            width,

        "height":
            height,

        "duration_sec":
            (
                frame_count
                /
                float(
                    fps
                )
                if fps > 0
                else None
            ),
    }


# ============================================================
# Logger
# ============================================================

class Logger:

    def __init__(
        self,
        path,
    ):

        self.path = Path(
            path
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


    def write(
        self,
        message="",
    ):

        message = str(
            message
        )

        print(
            message
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as f:

            f.write(
                message
                +
                "\n"
            )


# ============================================================
# Stage Execution
# ============================================================

def run_stage(
    name,
    command,
    expected_outputs,
    logger,
    dry_run=False,
):

    logger.write(
        ""
    )

    logger.write(
        "=" * 70
    )

    logger.write(
        f"STAGE: {name}"
    )

    logger.write(
        "=" * 70
    )

    logger.write(
        "COMMAND:"
    )

    command_text = (
        " ".join(
            str(x)
            for x
            in command
        )
    )

    logger.write(
        command_text
    )


    started_at = (
        now_iso()
    )


    if dry_run:

        return {
            "stage":
                name,

            "status":
                "dry_run",

            "started_at":
                started_at,

            "finished_at":
                now_iso(),

            "duration_sec":
                0.0,

            "command":
                [
                    str(x)
                    for x
                    in command
                ],

            "outputs":
                [
                    str(x)
                    for x
                    in expected_outputs
                ],
        }


    start = (
        time.perf_counter()
    )


    process = subprocess.Popen(
        [
            str(x)
            for x
            in command
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


    if process.stdout is None:

        raise RuntimeError(
            f"Could not capture output "
            f"for stage: {name}"
        )


    with logger.path.open(
        "a",
        encoding="utf-8",
    ) as logfile:

        for line in process.stdout:

            print(
                line,
                end="",
            )

            logfile.write(
                line
            )


    return_code = (
        process.wait()
    )


    duration = (
        time.perf_counter()
        -
        start
    )


    if return_code != 0:

        raise RuntimeError(
            f"Stage failed: "
            f"{name} "
            f"(exit={return_code})"
        )


    for output in (
        expected_outputs
    ):

        require_file(
            output,
            f"{name} output",
        )


    logger.write(
        f"SUCCESS: "
        f"{name} "
        f"({duration:.2f}s)"
    )


    return {
        "stage":
            name,

        "status":
            "success",

        "started_at":
            started_at,

        "finished_at":
            now_iso(),

        "duration_sec":
            round(
                duration,
                4,
            ),

        "command":
            [
                str(x)
                for x
                in command
            ],

        "outputs":
            [
                str(x)
                for x
                in expected_outputs
            ],
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Football CV Analytics "
            "Hybrid Pipeline V3"
        )
    )


    parser.add_argument(
        "--video",
        required=True,
        help=(
            "Input football video."
        ),
    )


    parser.add_argument(
        "--run-name",
        required=True,
        help=(
            "Name under runs/."
        ),
    )


    parser.add_argument(
        "--config",
        default=(
            "configs/default.yaml"
        ),
    )


    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Override config device "
            "(auto/mps/cpu/0/etc)."
        ),
    )


    parser.add_argument(
        "--resume",
        action="store_true",
    )


    parser.add_argument(
        "--overwrite",
        action="store_true",
    )


    parser.add_argument(
        "--skip-qa",
        action="store_true",
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
    )


    args = (
        parser.parse_args()
    )


    # ========================================================
    # Resolve Main Inputs
    # ========================================================

    video_path = resolve_path(
        args.video
    )


    config_path = resolve_path(
        args.config
    )


    require_file(
        video_path,
        "Input video",
    )


    require_file(
        config_path,
        "Config",
    )


    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        config = (
            yaml.safe_load(
                f
            )
        )


    video_info = (
        get_video_info(
            video_path
        )
    )


    fps = (
        video_info[
            "fps"
        ]
    )


    # ========================================================
    # Models
    # ========================================================

    models_cfg = (
        config[
            "models"
        ]
    )


    player_model = resolve_path(
        models_cfg[
            "player_detector"
        ]
    )


    pitch_model = resolve_path(
        models_cfg[
            "pitch_detector"
        ]
    )


    old_ball_model = resolve_path(
        models_cfg[
            "old_ball_detector"
        ]
    )


    forza_ball_model = resolve_path(
        models_cfg[
            "forza_ball_detector"
        ]
    )


    require_file(
        player_model,
        "Player model",
    )


    require_file(
        pitch_model,
        "Pitch model",
    )


    require_file(
        old_ball_model,
        "Old ball model",
    )


    require_file(
        forza_ball_model,
        "Forza ball model",
    )


    # ========================================================
    # Runtime
    # ========================================================

    if args.device is not None:

        device = (
            args.device
        )

    else:

        device = (
            config[
                "runtime"
            ].get(
                "device",
                "auto",
            )
        )


    python_exec = (
        sys.executable
    )


    # ========================================================
    # Current Legacy Middle Baselines
    # ========================================================

    baseline_cfg = (
        config[
            "pipeline"
        ][
            "baseline"
        ]
    )


    possession_v5_3_baseline = (
        resolve_path(
            baseline_cfg[
                "possession_v5_3"
            ]
        )
    )


    controller_gate_v2_baseline = (
        resolve_path(
            baseline_cfg[
                "controller_gate_v2"
            ]
        )
    )


    air_touch_v4_baseline = (
        resolve_path(
            baseline_cfg[
                "air_touch_v4"
            ]
        )
    )


    require_file(
        possession_v5_3_baseline,
        "Possession V5.3 baseline",
    )


    require_file(
        controller_gate_v2_baseline,
        "Controller Gate V2 baseline",
    )


    require_file(
        air_touch_v4_baseline,
        "AirTouch V4 baseline",
    )


    # ========================================================
    # Config Sections
    # ========================================================

    trajectory_cfg = (
        config[
            "pipeline"
        ][
            "trajectory"
        ]
    )


    motion_cfg = (
        config[
            "pipeline"
        ][
            "motion_thresholds"
        ]
    )


    # ========================================================
    # Run Directory
    # ========================================================

    run_dir = (
        ROOT
        /
        "runs"
        /
        args.run_name
    )


    if (
        run_dir.exists()
        and
        args.overwrite
    ):

        shutil.rmtree(
            run_dir
        )


    if (
        run_dir.exists()
        and
        not args.resume
        and
        not args.dry_run
    ):

        raise RuntimeError(
            f"Run already exists: "
            f"{run_dir}\n"
            f"Use --resume or --overwrite."
        )


    data_dir = (
        run_dir
        /
        "data"
    )


    json_dir = (
        run_dir
        /
        "json"
    )


    video_dir = (
        run_dir
        /
        "videos"
    )


    log_dir = (
        run_dir
        /
        "logs"
    )


    for directory in [
        data_dir,
        json_dir,
        video_dir,
        log_dir,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


    logger = Logger(
        log_dir
        /
        "pipeline.log"
    )


    # ========================================================
    # Config Snapshot
    # ========================================================

    snapshot_path = (
        run_dir
        /
        "config.snapshot.yaml"
    )


    shutil.copy2(
        config_path,
        snapshot_path,
    )


    # ========================================================
    # Player Branch Outputs
    # ========================================================

    tracking_raw = (
        data_dir
        /
        "tracking_data_v4.csv"
    )


    tracking_clean = (
        data_dir
        /
        "tracking_data_clean_v2.csv"
    )


    radar_video = (
        video_dir
        /
        "radar_v4.mp4"
    )


    # ========================================================
    # Ball Branch Outputs
    # ========================================================

    ball_raw = (
        data_dir
        /
        "ball_tracking_ensemble.csv"
    )


    ball_trusted = (
        data_dir
        /
        "ball_tracking_trusted.csv"
    )


    ball_qa_video = (
        video_dir
        /
        "ball_tracking_trusted_qa.mp4"
    )


    # ========================================================
    # Ball Motion Outputs
    # ========================================================

    motion_frames = (
        data_dir
        /
        "ball_motion_state_v1_frames.csv"
    )


    motion_episodes = (
        data_dir
        /
        "ball_motion_state_v1_episodes.csv"
    )


    motion_receptions = (
        data_dir
        /
        "ball_motion_state_v1_receptions.csv"
    )


    motion_dribbles = (
        data_dir
        /
        "ball_motion_state_v1_dribble_gaps.csv"
    )


    # ========================================================
    # Possession V6 Outputs
    # ========================================================

    possession_frames = (
        data_dir
        /
        "possession_v6_frames.csv"
    )


    possession_episodes = (
        data_dir
        /
        "possession_v6_episodes.csv"
    )


    # ========================================================
    # Event Outputs
    # ========================================================

    events_csv = (
        data_dir
        /
        "football_events_v1.csv"
    )


    events_json_source = (
        data_dir
        /
        "football_events_v1.json"
    )


    summary_json_source = (
        data_dir
        /
        "match_summary_v1.json"
    )


    event_qa_video = (
        video_dir
        /
        "football_event_qa_v1.mp4"
    )


    # ========================================================
    # Build Stage Graph
    # ========================================================

    stages = []


    # ========================================================
    # Stage 1
    #
    # Player tracking + pitch mapping
    # ========================================================

    stages.append(
        {
            "name":
                "radar_v4",

            "command":
                [
                    python_exec,

                    "src/radar_v4_cli.py",

                    "--video",
                    str(
                        video_path
                    ),

                    "--player-model",
                    str(
                        player_model
                    ),

                    "--pitch-model",
                    str(
                        pitch_model
                    ),

                    "--output-csv",
                    str(
                        tracking_raw
                    ),

                    "--output-video",
                    str(
                        radar_video
                    ),

                    "--device",
                    str(
                        device
                    ),
                ],

            "outputs":
                [
                    tracking_raw,
                    radar_video,
                ],
        }
    )


    # ========================================================
    # Stage 2
    #
    # Integrated trajectory cleaning
    # ========================================================

    stages.append(
        {
            "name":
                "trajectory_cleaning",

            "command":
                [
                    python_exec,

                    "src/trajectory_cleaning_cli.py",

                    "--input",
                    str(
                        tracking_raw
                    ),

                    "--output",
                    str(
                        tracking_clean
                    ),

                    "--fps",
                    str(
                        fps
                    ),

                    "--min-track-frames",
                    str(
                        trajectory_cfg[
                            "min_track_frames"
                        ]
                    ),

                    "--median-window",
                    str(
                        trajectory_cfg[
                            "median_window"
                        ]
                    ),

                    "--stage1-sg-window",
                    str(
                        trajectory_cfg[
                            "stage1_sg_window"
                        ]
                    ),

                    "--sg-polyorder",
                    str(
                        trajectory_cfg[
                            "sg_polyorder"
                        ]
                    ),

                    "--stage1-velocity-half-window",
                    str(
                        trajectory_cfg[
                            "stage1_velocity_half_window"
                        ]
                    ),

                    "--max-reasonable-speed",
                    str(
                        trajectory_cfg[
                            "max_reasonable_speed"
                        ]
                    ),

                    "--global-high-speed-threshold",
                    str(
                        trajectory_cfg[
                            "global_high_speed_threshold"
                        ]
                    ),

                    "--min-simultaneous-players",
                    str(
                        trajectory_cfg[
                            "min_simultaneous_players"
                        ]
                    ),

                    "--expand-bad-frame",
                    str(
                        trajectory_cfg[
                            "expand_bad_frame"
                        ]
                    ),

                    "--stage2-sg-window",
                    str(
                        trajectory_cfg[
                            "stage2_sg_window"
                        ]
                    ),

                    "--stage2-velocity-half-window",
                    str(
                        trajectory_cfg[
                            "stage2_velocity_half_window"
                        ]
                    ),
                ],

            "outputs":
                [
                    tracking_clean,
                ],
        }
    )


    # ========================================================
    # Stage 3
    #
    # Ball branch
    #
    # PASS A:
    # cross-model ensemble tracking
    #
    # PASS B:
    # trusted filtering + interpolation
    # ========================================================

    ball_command = [
        python_exec,

        "src/ball_tracking_cli.py",

        "--video",
        str(
            video_path
        ),

        "--old-ball-model",
        str(
            old_ball_model
        ),

        "--forza-model",
        str(
            forza_ball_model
        ),

        "--pitch-model",
        str(
            pitch_model
        ),

        "--raw-output-csv",
        str(
            ball_raw
        ),

        "--output-csv",
        str(
            ball_trusted
        ),

        "--device",
        str(
            device
        ),
    ]


    ball_outputs = [
        ball_raw,
        ball_trusted,
    ]


    if args.skip_qa:

        ball_command.append(
            "--skip-qa"
        )

    else:

        ball_command.extend(
            [
                "--output-video",
                str(
                    ball_qa_video
                ),
            ]
        )

        ball_outputs.append(
            ball_qa_video
        )


    stages.append(
        {
            "name":
                "ball_tracking",

            "command":
                ball_command,

            "outputs":
                ball_outputs,
        }
    )


    # ========================================================
    # MIDDLE DEPENDENCY GAP
    #
    # Player + Ball perception are now run-local.
    #
    # These are still legacy baseline inputs:
    #
    # possession_v5_3
    # controller_gate_v2
    # air_touch_v4
    #
    # They will be replaced in the next productionization
    # phase.
    # ========================================================


    # ========================================================
    # Stage 4
    #
    # Ball Motion State
    # ========================================================

    stages.append(
        {
            "name":
                "ball_motion_state_v1",

            "command":
                [
                    python_exec,

                    "src/ball_motion_state_v1.py",

                    "--possession",
                    str(
                        possession_v5_3_baseline
                    ),

                    "--gate",
                    str(
                        controller_gate_v2_baseline
                    ),

                    "--events",
                    str(
                        air_touch_v4_baseline
                    ),

                    "--output-dir",
                    str(
                        data_dir
                    ),

                    "--launch-min-speed",
                    str(
                        motion_cfg[
                            "launch_min_speed_mps"
                        ]
                    ),

                    "--reception-confirm-sec",
                    str(
                        motion_cfg[
                            "reception_confirm_sec"
                        ]
                    ),

                    "--dribble-return-sec",
                    str(
                        motion_cfg[
                            "dribble_return_sec"
                        ]
                    ),
                ],

            "outputs":
                [
                    motion_frames,
                    motion_episodes,
                    motion_receptions,
                    motion_dribbles,
                ],
        }
    )


    # ========================================================
    # Stage 5
    #
    # Possession V6
    # ========================================================

    stages.append(
        {
            "name":
                "possession_v6",

            "command":
                [
                    python_exec,

                    "src/possession_v6.py",

                    "--motion",
                    str(
                        motion_frames
                    ),

                    "--output-dir",
                    str(
                        data_dir
                    ),
                ],

            "outputs":
                [
                    possession_frames,
                    possession_episodes,
                ],
        }
    )


    # ========================================================
    # Stage 6
    #
    # Football Event Engine
    # ========================================================

    stages.append(
        {
            "name":
                "football_event_engine_v1",

            "command":
                [
                    python_exec,

                    "src/football_event_engine_v1.py",

                    "--frames",
                    str(
                        possession_frames
                    ),

                    "--possession-episodes",
                    str(
                        possession_episodes
                    ),

                    "--motion-episodes",
                    str(
                        motion_episodes
                    ),

                    "--air-events",
                    str(
                        air_touch_v4_baseline
                    ),

                    "--dribble-gaps",
                    str(
                        motion_dribbles
                    ),

                    "--output-dir",
                    str(
                        data_dir
                    ),
                ],

            "outputs":
                [
                    events_csv,
                    events_json_source,
                    summary_json_source,
                ],
        }
    )


    # ========================================================
    # Stage 7
    #
    # Football Event QA
    # ========================================================

    if not args.skip_qa:

        stages.append(
            {
                "name":
                    "football_event_qa_v1",

                "command":
                    [
                        python_exec,

                        "src/football_event_qa_v1.py",

                        "--video",
                        str(
                            video_path
                        ),

                        "--frames",
                        str(
                            possession_frames
                        ),

                        "--events",
                        str(
                            events_csv
                        ),

                        "--output",
                        str(
                            event_qa_video
                        ),
                    ],

                "outputs":
                    [
                        event_qa_video,
                    ],
            }
        )


    # ========================================================
    # Manifest
    # ========================================================

    manifest = {
        "project":
            config[
                "project"
            ][
                "name"
            ],

        "runner_version":
            "3.0.0",

        "pipeline_scope":
            "hybrid_v3",

        "run_name":
            args.run_name,

        "status":
            "running",

        "started_at":
            now_iso(),

        "video":
            str(
                video_path
            ),

        "video_info":
            video_info,

        "device":
            str(
                device
            ),

        "config":
            str(
                config_path
            ),

        "config_sha256":
            sha256_file(
                config_path
            ),

        "git_commit":
            get_git_commit(),

        "python_version":
            sys.version,

        "platform":
            platform.platform(),

        "models":
            {
                "player_detector":
                    str(
                        player_model
                    ),

                "pitch_detector":
                    str(
                        pitch_model
                    ),

                "old_ball_detector":
                    str(
                        old_ball_model
                    ),

                "forza_ball_detector":
                    str(
                        forza_ball_model
                    ),
            },

        "run_local_perception":
            {
                "player_tracking":
                    str(
                        tracking_raw
                    ),

                "player_tracking_clean":
                    str(
                        tracking_clean
                    ),

                "ball_tracking_raw":
                    str(
                        ball_raw
                    ),

                "ball_tracking_trusted":
                    str(
                        ball_trusted
                    ),
            },

        "legacy_middle_baselines":
            {
                "possession_v5_3":
                    str(
                        possession_v5_3_baseline
                    ),

                "controller_gate_v2":
                    str(
                        controller_gate_v2_baseline
                    ),

                "air_touch_v4":
                    str(
                        air_touch_v4_baseline
                    ),
            },

        "dependency_gap":
            (
                "Player and ball perception branches "
                "are run-local. "
                "Middle possession, air-interaction, "
                "and controller reasoning stages "
                "still use validated legacy baseline "
                "outputs."
            ),

        "stages":
            [],
    }


    # ========================================================
    # Start Logging
    # ========================================================

    logger.write(
        f"RUN: "
        f"{args.run_name}"
    )


    logger.write(
        f"VIDEO: "
        f"{video_path}"
    )


    logger.write(
        f"VIDEO FPS: "
        f"{fps:.3f}"
    )


    logger.write(
        "PIPELINE SCOPE: hybrid_v3"
    )


    logger.write(
        "PLAYER BRANCH: run-local"
    )


    logger.write(
        "BALL BRANCH: run-local"
    )


    logger.write(
        "WARNING: middle pipeline still uses "
        "validated legacy baseline outputs."
    )


    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # Execute Pipeline
    # ========================================================

    try:

        for stage in stages:

            outputs = (
                stage[
                    "outputs"
                ]
            )


            # ------------------------------------------------
            # Resume
            # ------------------------------------------------

            if (
                args.resume
                and
                all(
                    Path(
                        output
                    ).exists()
                    and
                    Path(
                        output
                    ).stat().st_size
                    >
                    0
                    for output
                    in outputs
                )
            ):

                logger.write(
                    ""
                )

                logger.write(
                    f"SKIP: "
                    f"{stage['name']} "
                    f"(existing outputs)"
                )


                manifest[
                    "stages"
                ].append(
                    {
                        "stage":
                            stage[
                                "name"
                            ],

                        "status":
                            "skipped_existing",

                        "duration_sec":
                            0.0,

                        "outputs":
                            [
                                str(x)
                                for x
                                in outputs
                            ],
                    }
                )


                continue


            result = run_stage(
                name=(
                    stage[
                        "name"
                    ]
                ),

                command=(
                    stage[
                        "command"
                    ]
                ),

                expected_outputs=(
                    outputs
                ),

                logger=logger,

                dry_run=(
                    args.dry_run
                ),
            )


            manifest[
                "stages"
            ].append(
                result
            )


        # ====================================================
        # Product-facing JSON
        # ====================================================

        if not args.dry_run:

            shutil.copy2(
                events_json_source,
                json_dir
                /
                "events.json",
            )


            shutil.copy2(
                summary_json_source,
                json_dir
                /
                "summary.json",
            )


        manifest[
            "status"
        ] = (
            "dry_run"
            if args.dry_run
            else "success"
        )


    except Exception as exc:

        manifest[
            "status"
        ] = (
            "failed"
        )


        manifest[
            "error"
        ] = str(
            exc
        )


        logger.write(
            ""
        )


        logger.write(
            f"FAILED: "
            f"{exc}"
        )


        raise


    finally:

        total_runtime = (
            time.perf_counter()
            -
            total_start
        )


        manifest[
            "finished_at"
        ] = (
            now_iso()
        )


        manifest[
            "total_runtime_sec"
        ] = round(
            total_runtime,
            4,
        )


        manifest_path = (
            json_dir
            /
            "manifest.json"
        )


        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                manifest,
                f,
                ensure_ascii=False,
                indent=2,
            )


        runtime = {
            "run_name":
                args.run_name,

            "pipeline_scope":
                "hybrid_v3",

            "total_runtime_sec":
                round(
                    total_runtime,
                    4,
                ),

            "stage_runtime_sec":
                {
                    stage.get(
                        "stage"
                    ):
                        stage.get(
                            "duration_sec",
                            0.0,
                        )

                    for stage
                    in manifest[
                        "stages"
                    ]
                },
        }


        with (
            run_dir
            /
            "runtime.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                runtime,
                f,
                indent=2,
            )


    # ========================================================
    # Complete
    # ========================================================

    logger.write(
        ""
    )


    logger.write(
        "=" * 70
    )


    logger.write(
        "PIPELINE COMPLETE"
    )


    logger.write(
        "=" * 70
    )


    logger.write(
        f"Run: "
        f"{run_dir}"
    )


    logger.write(
        ""
    )


    logger.write(
        "PLAYER BRANCH"
    )


    logger.write(
        f"Raw tracking: "
        f"{tracking_raw}"
    )


    logger.write(
        f"Clean tracking: "
        f"{tracking_clean}"
    )


    logger.write(
        f"Radar video: "
        f"{radar_video}"
    )


    logger.write(
        ""
    )


    logger.write(
        "BALL BRANCH"
    )


    logger.write(
        f"Raw ball: "
        f"{ball_raw}"
    )


    logger.write(
        f"Trusted ball: "
        f"{ball_trusted}"
    )


    if not args.skip_qa:

        logger.write(
            f"Ball QA: "
            f"{ball_qa_video}"
        )


    logger.write(
        ""
    )


    logger.write(
        "EVENT OUTPUT"
    )


    logger.write(
        f"Events CSV: "
        f"{events_csv}"
    )


    logger.write(
        f"Events JSON: "
        f"{json_dir / 'events.json'}"
    )


    logger.write(
        f"Summary JSON: "
        f"{json_dir / 'summary.json'}"
    )


    logger.write(
        f"Manifest: "
        f"{json_dir / 'manifest.json'}"
    )


    if not args.skip_qa:

        logger.write(
            f"Event QA: "
            f"{event_qa_video}"
        )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()
