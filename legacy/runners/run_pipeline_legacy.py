#!/usr/bin/env python3

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def get_git_commit():
    try:
        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    except Exception:
        return None


def require_file(
    path,
    label,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if (
        path.is_file()
        and
        path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"{label} is empty: {path}"
        )


class Logger:

    def __init__(
        self,
        path,
    ):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write(
        self,
        message="",
    ):
        message = str(message)

        print(message)

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                message + "\n"
            )


def run_stage(
    name,
    command,
    expected_outputs,
    logger,
    dry_run=False,
):

    start_iso = now_iso()

    start_time = (
        time.perf_counter()
    )

    logger.write("")
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

    logger.write(
        " ".join(
            str(x)
            for x in command
        )
    )


    if dry_run:

        return {
            "stage":
                name,

            "status":
                "dry_run",

            "started_at":
                start_iso,

            "finished_at":
                now_iso(),

            "duration_sec":
                0.0,

            "outputs":
                [
                    str(x)
                    for x
                    in expected_outputs
                ],
        }


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


    with logger.path.open(
        "a",
        encoding="utf-8",
    ) as logfile:

        for line in process.stdout:

            line = line.rstrip(
                "\n"
            )

            print(line)

            logfile.write(
                line + "\n"
            )


    return_code = (
        process.wait()
    )


    duration = (
        time.perf_counter()
        -
        start_time
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
        f"SUCCESS: {name} "
        f"({duration:.2f}s)"
    )


    return {
        "stage":
            name,

        "status":
            "success",

        "started_at":
            start_iso,

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


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Football CV "
            "pipeline runner"
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
        default=(
            "configs/default.yaml"
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


    args = parser.parse_args()


    # =====================================================
    # PATHS
    # =====================================================

    video_path = (
        ROOT
        /
        args.video
    ).resolve()


    config_path = (
        ROOT
        /
        args.config
    ).resolve()


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
            yaml.safe_load(f)
        )


    run_root = (
        ROOT
        /
        "runs"
        /
        args.run_name
    )


    if (
        run_root.exists()
        and
        args.overwrite
    ):

        shutil.rmtree(
            run_root
        )


    if (
        run_root.exists()
        and
        not args.resume
        and
        not args.dry_run
    ):

        raise RuntimeError(
            f"Run already exists: "
            f"{run_root}\n"
            f"Use --resume or --overwrite."
        )


    data_dir = (
        run_root
        /
        "data"
    )


    json_dir = (
        run_root
        /
        "json"
    )


    video_dir = (
        run_root
        /
        "videos"
    )


    log_dir = (
        run_root
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


    # =====================================================
    # CONFIG SNAPSHOT
    # =====================================================

    shutil.copy2(
        config_path,
        run_root
        /
        "config.snapshot.yaml",
    )


    # =====================================================
    # CURRENT VALIDATED BASELINE INPUTS
    #
    # NOTE:
    # Upstream scripts are still legacy/hard-coded.
    # We will refactor them next.
    # =====================================================

    baseline = (
        config[
            "pipeline"
        ][
            "baseline"
        ]
    )


    possession_v5_3 = (
        ROOT
        /
        baseline[
            "possession_v5_3"
        ]
    ).resolve()


    controller_gate_v2 = (
        ROOT
        /
        baseline[
            "controller_gate_v2"
        ]
    ).resolve()


    air_touch_v4 = (
        ROOT
        /
        baseline[
            "air_touch_v4"
        ]
    ).resolve()


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


    thresholds = (
        config[
            "pipeline"
        ][
            "motion_thresholds"
        ]
    )


    python_exec = (
        sys.executable
    )


    # =====================================================
    # OUTPUT FILES
    # =====================================================

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


    qa_video = (
        video_dir
        /
        "football_event_qa_v1.mp4"
    )


    # =====================================================
    # PIPELINE STAGES
    # =====================================================

    stages = []


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
                        possession_v5_3
                    ),

                    "--gate",
                    str(
                        controller_gate_v2
                    ),

                    "--events",
                    str(
                        air_touch_v4
                    ),

                    "--output-dir",
                    str(
                        data_dir
                    ),

                    "--launch-min-speed",
                    str(
                        thresholds[
                            "launch_min_speed_mps"
                        ]
                    ),

                    "--reception-confirm-sec",
                    str(
                        thresholds[
                            "reception_confirm_sec"
                        ]
                    ),

                    "--dribble-return-sec",
                    str(
                        thresholds[
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
                        air_touch_v4
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
                            qa_video
                        ),
                    ],

                "outputs":
                    [
                        qa_video
                    ],
            }
        )


    # =====================================================
    # MANIFEST
    # =====================================================

    manifest = {
        "project":
            config[
                "project"
            ][
                "name"
            ],

        "runner_version":
            "1.0.0",

        "scope":
            "analysis_tail",

        "run_name":
            args.run_name,

        "started_at":
            now_iso(),

        "video":
            str(
                video_path
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

        "baseline_inputs":
            {
                "possession_v5_3":
                    str(
                        possession_v5_3
                    ),

                "controller_gate_v2":
                    str(
                        controller_gate_v2
                    ),

                "air_touch_v4":
                    str(
                        air_touch_v4
                    ),
            },

        "stages":
            [],
    }


    logger.write(
        f"RUN: {args.run_name}"
    )


    logger.write(
        f"VIDEO: {video_path}"
    )


    logger.write(
        "PIPELINE SCOPE: analysis_tail"
    )


    total_start = (
        time.perf_counter()
    )


    try:

        for stage in stages:

            outputs = (
                stage[
                    "outputs"
                ]
            )


            if (
                args.resume

                and

                all(
                    output.exists()
                    and
                    output.stat().st_size > 0

                    for output
                    in outputs
                )
            ):

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


        # =================================================
        # PRODUCT-FACING JSON
        # =================================================

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
        ] = "failed"


        manifest[
            "error"
        ] = str(exc)


        logger.write(
            f"FAILED: {exc}"
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
        ] = now_iso()


        manifest[
            "total_runtime_sec"
        ] = round(
            total_runtime,
            4,
        )


        with (
            json_dir
            /
            "manifest.json"
        ).open(
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
            run_root
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


    logger.write("")
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
        f"Run directory: "
        f"{run_root}"
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
            f"QA video: "
            f"{qa_video}"
        )


if __name__ == "__main__":
    main()
