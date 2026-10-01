#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return (ROOT / path).resolve()


def require_file(path, name):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )


def get_video_fps(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    cap.release()

    if fps <= 0:
        raise RuntimeError(
            "Invalid video FPS."
        )

    return float(fps)


def run_stage(
    name,
    command,
    outputs,
    log_file,
    dry_run=False,
):

    print("")
    print("=" * 70)
    print(f"STAGE: {name}")
    print("=" * 70)

    command_text = " ".join(
        str(x)
        for x in command
    )

    print("COMMAND:")
    print(command_text)

    with open(
        log_file,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            f"\n{'=' * 70}\n"
        )

        f.write(
            f"STAGE: {name}\n"
        )

        f.write(
            command_text + "\n"
        )

    if dry_run:

        return {
            "stage": name,
            "status": "dry_run",
            "duration_sec": 0.0,
        }

    start = time.perf_counter()

    process = subprocess.Popen(
        [
            str(x)
            for x in command
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert process.stdout is not None

    with open(
        log_file,
        "a",
        encoding="utf-8",
    ) as f:

        for line in process.stdout:

            print(
                line,
                end="",
            )

            f.write(
                line
            )

    code = process.wait()

    duration = (
        time.perf_counter()
        -
        start
    )

    if code != 0:

        raise RuntimeError(
            f"{name} failed "
            f"(exit={code})"
        )

    for output in outputs:

        require_file(
            output,
            f"{name} output",
        )

    print(
        f"SUCCESS: {name} "
        f"({duration:.2f}s)"
    )

    return {
        "stage": name,
        "status": "success",
        "duration_sec": round(
            duration,
            3,
        ),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Football CV Hybrid Pipeline V2"
        )
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--skip-qa",
        action="store_true",
    )

    args = parser.parse_args()


    # ========================================================
    # Load Config
    # ========================================================

    video_path = resolve_path(
        args.video
    )

    config_path = resolve_path(
        args.config
    )

    require_file(
        video_path,
        "Video",
    )

    require_file(
        config_path,
        "Config",
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(
            f
        )


    fps = get_video_fps(
        video_path
    )


    # ========================================================
    # Models
    # ========================================================

    player_model = resolve_path(
        config[
            "models"
        ][
            "player_detector"
        ]
    )

    pitch_model = resolve_path(
        config[
            "models"
        ][
            "pitch_detector"
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


    device = (
        args.device
        if args.device is not None
        else
        config[
            "runtime"
        ].get(
            "device",
            "auto",
        )
    )


    # ========================================================
    # Legacy middle baseline
    # ========================================================

    baseline = (
        config[
            "pipeline"
        ][
            "baseline"
        ]
    )

    possession_v5_3 = resolve_path(
        baseline[
            "possession_v5_3"
        ]
    )

    controller_gate_v2 = resolve_path(
        baseline[
            "controller_gate_v2"
        ]
    )

    air_touch_v4 = resolve_path(
        baseline[
            "air_touch_v4"
        ]
    )

    require_file(
        possession_v5_3,
        "Possession V5.3 baseline",
    )

    require_file(
        controller_gate_v2,
        "Controller Gate V2 baseline",
    )

    require_file(
        air_touch_v4,
        "AirTouch V4 baseline",
    )


    # ========================================================
    # Run directory
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
            f"Run exists: {run_dir}\n"
            "Use --resume or --overwrite"
        )


    data_dir = (
        run_dir
        /
        "data"
    )

    video_dir = (
        run_dir
        /
        "videos"
    )

    json_dir = (
        run_dir
        /
        "json"
    )

    log_dir = (
        run_dir
        /
        "logs"
    )


    for folder in [
        data_dir,
        video_dir,
        json_dir,
        log_dir,
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


    log_file = (
        log_dir
        /
        "pipeline.log"
    )


    # ========================================================
    # Output paths
    # ========================================================

    tracking_raw = (
        data_dir
        /
        "tracking_data_v4.csv"
    )

    radar_video = (
        video_dir
        /
        "radar_v4.mp4"
    )

    tracking_clean = (
        data_dir
        /
        "tracking_data_clean_v2.csv"
    )


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


    events_csv = (
        data_dir
        /
        "football_events_v1.csv"
    )

    events_json = (
        data_dir
        /
        "football_events_v1.json"
    )

    summary_json = (
        data_dir
        /
        "match_summary_v1.json"
    )


    qa_video = (
        video_dir
        /
        "football_event_qa_v1.mp4"
    )


    python_exec = (
        sys.executable
    )

    trajectory = (
        config[
            "pipeline"
        ][
            "trajectory"
        ]
    )

    motion = (
        config[
            "pipeline"
        ][
            "motion_thresholds"
        ]
    )


    # ========================================================
    # Stage definitions
    # ========================================================

    stages = []


    # --------------------------------------------------------
    # 1. Radar
    # --------------------------------------------------------

    stages.append(
        {
            "name": "radar_v4",

            "command": [
                python_exec,
                "src/radar_v4_cli.py",

                "--video",
                str(video_path),

                "--player-model",
                str(player_model),

                "--pitch-model",
                str(pitch_model),

                "--output-csv",
                str(tracking_raw),

                "--output-video",
                str(radar_video),

                "--device",
                str(device),
            ],

            "outputs": [
                tracking_raw,
                radar_video,
            ],
        }
    )


    # --------------------------------------------------------
    # 2. Trajectory Cleaning
    # --------------------------------------------------------

    stages.append(
        {
            "name":
                "trajectory_cleaning",

            "command": [
                python_exec,
                "src/trajectory_cleaning_cli.py",

                "--input",
                str(tracking_raw),

                "--output",
                str(tracking_clean),

                "--fps",
                str(fps),

                "--min-track-frames",
                str(
                    trajectory[
                        "min_track_frames"
                    ]
                ),

                "--median-window",
                str(
                    trajectory[
                        "median_window"
                    ]
                ),

                "--stage1-sg-window",
                str(
                    trajectory[
                        "stage1_sg_window"
                    ]
                ),

                "--sg-polyorder",
                str(
                    trajectory[
                        "sg_polyorder"
                    ]
                ),

                "--stage1-velocity-half-window",
                str(
                    trajectory[
                        "stage1_velocity_half_window"
                    ]
                ),

                "--max-reasonable-speed",
                str(
                    trajectory[
                        "max_reasonable_speed"
                    ]
                ),

                "--global-high-speed-threshold",
                str(
                    trajectory[
                        "global_high_speed_threshold"
                    ]
                ),

                "--min-simultaneous-players",
                str(
                    trajectory[
                        "min_simultaneous_players"
                    ]
                ),

                "--expand-bad-frame",
                str(
                    trajectory[
                        "expand_bad_frame"
                    ]
                ),

                "--stage2-sg-window",
                str(
                    trajectory[
                        "stage2_sg_window"
                    ]
                ),

                "--stage2-velocity-half-window",
                str(
                    trajectory[
                        "stage2_velocity_half_window"
                    ]
                ),
            ],

            "outputs": [
                tracking_clean,
            ],
        }
    )


    # --------------------------------------------------------
    # Middle dependency gap
    #
    # 尚未 CLI 化：
    # possession v4 / air touch / controller / possession v5.x
    # --------------------------------------------------------


    # --------------------------------------------------------
    # 3. Ball Motion
    # --------------------------------------------------------

    stages.append(
        {
            "name":
                "ball_motion_state_v1",

            "command": [
                python_exec,
                "src/ball_motion_state_v1.py",

                "--possession",
                str(possession_v5_3),

                "--gate",
                str(controller_gate_v2),

                "--events",
                str(air_touch_v4),

                "--output-dir",
                str(data_dir),

                "--launch-min-speed",
                str(
                    motion[
                        "launch_min_speed_mps"
                    ]
                ),

                "--reception-confirm-sec",
                str(
                    motion[
                        "reception_confirm_sec"
                    ]
                ),

                "--dribble-return-sec",
                str(
                    motion[
                        "dribble_return_sec"
                    ]
                ),
            ],

            "outputs": [
                motion_frames,
                motion_episodes,
                motion_receptions,
                motion_dribbles,
            ],
        }
    )


    # --------------------------------------------------------
    # 4. Possession V6
    # --------------------------------------------------------

    stages.append(
        {
            "name":
                "possession_v6",

            "command": [
                python_exec,
                "src/possession_v6.py",

                "--motion",
                str(motion_frames),

                "--output-dir",
                str(data_dir),
            ],

            "outputs": [
                possession_frames,
                possession_episodes,
            ],
        }
    )


    # --------------------------------------------------------
    # 5. Event Engine
    # --------------------------------------------------------

    stages.append(
        {
            "name":
                "football_event_engine_v1",

            "command": [
                python_exec,
                "src/football_event_engine_v1.py",

                "--frames",
                str(possession_frames),

                "--possession-episodes",
                str(possession_episodes),

                "--motion-episodes",
                str(motion_episodes),

                "--air-events",
                str(air_touch_v4),

                "--dribble-gaps",
                str(motion_dribbles),

                "--output-dir",
                str(data_dir),
            ],

            "outputs": [
                events_csv,
                events_json,
                summary_json,
            ],
        }
    )


    # --------------------------------------------------------
    # 6. Event QA
    # --------------------------------------------------------

    if not args.skip_qa:

        stages.append(
            {
                "name":
                    "football_event_qa_v1",

                "command": [
                    python_exec,
                    "src/football_event_qa_v1.py",

                    "--video",
                    str(video_path),

                    "--frames",
                    str(possession_frames),

                    "--events",
                    str(events_csv),

                    "--output",
                    str(qa_video),
                ],

                "outputs": [
                    qa_video,
                ],
            }
        )


    # ========================================================
    # Execute
    # ========================================================

    print(
        f"RUN: {args.run_name}"
    )

    print(
        f"VIDEO: {video_path}"
    )

    print(
        f"VIDEO FPS: {fps:.3f}"
    )

    print(
        "PIPELINE SCOPE: hybrid_v2"
    )

    print(
        "WARNING: middle pipeline still "
        "uses validated legacy baseline outputs."
    )


    records = []

    total_start = (
        time.perf_counter()
    )


    for stage in stages:

        if (
            args.resume
            and
            all(
                Path(x).exists()
                and
                Path(x).stat().st_size > 0

                for x
                in stage[
                    "outputs"
                ]
            )
        ):

            print(
                f"SKIP: "
                f"{stage['name']}"
            )

            records.append(
                {
                    "stage":
                        stage[
                            "name"
                        ],

                    "status":
                        "skipped",
                }
            )

            continue


        record = run_stage(
            stage[
                "name"
            ],
            stage[
                "command"
            ],
            stage[
                "outputs"
            ],
            log_file,
            dry_run=(
                args.dry_run
            ),
        )

        records.append(
            record
        )


    total_runtime = (
        time.perf_counter()
        -
        total_start
    )


    # ========================================================
    # Product-facing outputs
    # ========================================================

    if not args.dry_run:

        shutil.copy2(
            events_json,
            json_dir
            /
            "events.json",
        )

        shutil.copy2(
            summary_json,
            json_dir
            /
            "summary.json",
        )


    manifest = {
        "pipeline_version":
            "hybrid_v2",

        "run_name":
            args.run_name,

        "video":
            str(video_path),

        "video_fps":
            fps,

        "device":
            device,

        "total_runtime_sec":
            round(
                total_runtime,
                3,
            ),

        "dependency_gap":
            (
                "Middle possession / "
                "controller / air-touch "
                "pipeline still uses "
                "legacy baseline outputs."
            ),

        "stages":
            records,
    }


    with open(
        json_dir
        /
        "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )


    with open(
        run_dir
        /
        "runtime.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "total_runtime_sec":
                    round(
                        total_runtime,
                        3,
                    ),

                "stages":
                    records,
            },
            f,
            indent=2,
        )


    print("")
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Run: {run_dir}"
    )

    print(
        f"Tracking raw: "
        f"{tracking_raw}"
    )

    print(
        f"Tracking clean: "
        f"{tracking_clean}"
    )

    print(
        f"Events: "
        f"{events_csv}"
    )


if __name__ == "__main__":
    main()
