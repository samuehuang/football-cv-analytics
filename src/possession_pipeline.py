#!/usr/bin/env python3

import argparse
import json
import shlex
import shutil
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# PROJECT
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

PYTHON = sys.executable


# ============================================================
# HELPERS
# ============================================================

def resolve_input(
    value,
    label,
):
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    if path.stat().st_size == 0:
        raise RuntimeError(
            f"{label} is empty: {path}"
        )

    return path


def resolve_run_dir(
    value,
):
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def require_script(
    filename,
):
    path = SRC_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Required canonical stage missing: {path}"
        )

    return path


def output_ready(
    path,
):
    return (
        path.is_file()
        and
        path.stat().st_size > 0
    )


def run_stage(
    name,
    command,
    outputs,
    log_path,
    resume=False,
    dry_run=False,
):
    print("")
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        shlex.join(
            [
                str(x)
                for x in command
            ]
        )
    )

    if (
        resume
        and
        outputs
        and
        all(
            output_ready(path)
            for path in outputs
        )
    ):
        print("")
        print(
            "SKIP — outputs already exist"
        )
        return

    if dry_run:
        print("")
        print(
            "DRY RUN — not executed"
        )
        return

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        process = subprocess.Popen(
            [
                str(x)
                for x in command
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(
                line,
                end="",
            )

            log_file.write(
                line
            )

        return_code = (
            process.wait()
        )

    if return_code != 0:
        raise RuntimeError(
            f"{name} failed "
            f"with exit code {return_code}"
        )

    missing = [
        path
        for path in outputs
        if not output_ready(path)
    ]

    if missing:
        raise RuntimeError(
            f"{name} completed but expected "
            f"outputs are missing:\n"
            +
            "\n".join(
                str(path)
                for path in missing
            )
        )


def write_manifest(
    path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Canonical football possession pipeline. "
            "Runs all validated possession reasoning stages "
            "from tracking inputs to final possession output."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--player-model",
        required=True,
    )

    parser.add_argument(
        "--players",
        required=True,
        help=(
            "Clean player trajectory CSV"
        ),
    )

    parser.add_argument(
        "--ball",
        required=True,
        help=(
            "Clean/trusted ball tracking CSV"
        ),
    )

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
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

    parser.add_argument(
        "--skip-qa",
        action="store_true",
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
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()


    if (
        args.resume
        and
        args.overwrite
    ):
        raise ValueError(
            "--resume and --overwrite "
            "cannot be used together"
        )


    # ========================================================
    # INPUTS
    # ========================================================

    video = resolve_input(
        args.video,
        "Video",
    )

    player_model = resolve_input(
        args.player_model,
        "Player model",
    )

    players = resolve_input(
        args.players,
        "Player tracking CSV",
    )

    ball = resolve_input(
        args.ball,
        "Ball tracking CSV",
    )


    # ========================================================
    # CANONICAL STAGES
    # ========================================================

    possession_seed_script = (
        require_script(
            "possession_seed_cli.py"
        )
    )

    air_interaction_script = (
        require_script(
            "air_interaction_cli.py"
        )
    )

    possession_v5_1_script = (
        require_script(
            "possession_v5_1_cli.py"
        )
    )

    controller_gate_script = (
        require_script(
            "controller_gate_cli.py"
        )
    )

    refinement_script = (
        require_script(
            "possession_refinement_cli.py"
        )
    )

    ball_motion_script = (
        require_script(
            "ball_motion_state_v1.py"
        )
    )

    possession_v6_script = (
        require_script(
            "possession_v6.py"
        )
    )


    # ========================================================
    # RUN DIRECTORY
    # ========================================================

    run_dir = resolve_run_dir(
        args.run_dir
    )

    if (
        args.overwrite
        and
        run_dir.exists()
        and
        not args.dry_run
    ):
        shutil.rmtree(
            run_dir
        )

    if (
        run_dir.exists()
        and
        not args.resume
        and
        not args.overwrite
        and
        any(
            run_dir.iterdir()
        )
    ):
        raise RuntimeError(
            f"Run directory already contains files: "
            f"{run_dir}\n"
            f"Use --resume or --overwrite."
        )


    data_dir = (
        run_dir
        /
        "data"
    )

    videos_dir = (
        run_dir
        /
        "videos"
    )

    logs_dir = (
        run_dir
        /
        "logs"
    )


    if not args.dry_run:

        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        videos_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        logs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    seed_frames = (
        data_dir
        /
        "possession_seed_frames.csv"
    )

    seed_summary = (
        data_dir
        /
        "possession_seed_summary.csv"
    )

    seed_intermediate = (
        data_dir
        /
        "possession_seed_geometry.csv"
    )


    # --------------------------------------------------------
    # Air interaction
    # --------------------------------------------------------

    air_interactions = (
        data_dir
        /
        "air_interaction_v3_interactions.csv"
    )

    air_v3_events = (
        data_dir
        /
        "air_interaction_v3_events.csv"
    )

    air_v3_confirmed = (
        data_dir
        /
        "air_interaction_v3_confirmed.csv"
    )

    air_events = (
        data_dir
        /
        "air_interaction_events.csv"
    )

    air_confirmed = (
        data_dir
        /
        "air_interaction_confirmed_air_touch.csv"
    )

    air_contests = (
        data_dir
        /
        "air_interaction_aerial_contests.csv"
    )

    air_qa_video = (
        videos_dir
        /
        "air_interaction_qa.mp4"
    )


    # --------------------------------------------------------
    # V5.1 integration
    # --------------------------------------------------------

    v5_1_frames = (
        data_dir
        /
        "possession_v5_1_frames.csv"
    )

    v5_1_summary = (
        data_dir
        /
        "possession_v5_1_summary.csv"
    )

    v5_1_audit = (
        data_dir
        /
        "possession_v5_1_audit.csv"
    )


    # --------------------------------------------------------
    # Controller Gate
    # --------------------------------------------------------

    gate_frames = (
        data_dir
        /
        "controller_gate.csv"
    )

    gate_flagged = (
        data_dir
        /
        "controller_gate_flagged.csv"
    )

    gate_qa_video = (
        videos_dir
        /
        "controller_gate_qa.mp4"
    )


    # --------------------------------------------------------
    # Refinement V5.2 / V5.3
    # --------------------------------------------------------

    v5_2_frames = (
        data_dir
        /
        "possession_v5_2_frames.csv"
    )

    v5_2_summary = (
        data_dir
        /
        "possession_v5_2_summary.csv"
    )

    v5_2_audit = (
        data_dir
        /
        "possession_v5_2_audit.csv"
    )

    v5_3_frames = (
        data_dir
        /
        "possession_v5_3_frames.csv"
    )

    v5_3_summary = (
        data_dir
        /
        "possession_v5_3_summary.csv"
    )

    v5_3_audit = (
        data_dir
        /
        "possession_v5_3_audit.csv"
    )

    contest_episodes = (
        data_dir
        /
        "possession_v5_3_contest_episodes.csv"
    )


    # --------------------------------------------------------
    # Motion
    # --------------------------------------------------------

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

    motion_dribble_gaps = (
        data_dir
        /
        "ball_motion_state_v1_dribble_gaps.csv"
    )


    # --------------------------------------------------------
    # V6
    # --------------------------------------------------------

    v6_frames = (
        data_dir
        /
        "possession_v6_frames.csv"
    )

    v6_episodes = (
        data_dir
        /
        "possession_v6_episodes.csv"
    )


    # --------------------------------------------------------
    # Production-facing canonical aliases
    # --------------------------------------------------------

    final_frames = (
        data_dir
        /
        "possession_frames.csv"
    )

    final_episodes = (
        data_dir
        /
        "possession_episodes.csv"
    )


    # ========================================================
    # MANIFEST
    # ========================================================

    manifest_path = (
        run_dir
        /
        "manifest.json"
    )

    manifest = {
        "pipeline":
            "possession_pipeline",

        "pipeline_version":
            "canonical_v1",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "inputs": {
            "video":
                str(video),

            "player_model":
                str(player_model),

            "players":
                str(players),

            "ball":
                str(ball),
        },

        "parameters": {
            "device":
                args.device,

            "fps":
                args.fps,

            "launch_min_speed":
                args.launch_min_speed,

            "reception_confirm_sec":
                args.reception_confirm_sec,

            "dribble_return_sec":
                args.dribble_return_sec,

            "skip_qa":
                args.skip_qa,
        },

        "final_outputs": {
            "frames":
                str(final_frames),

            "episodes":
                str(final_episodes),
        },
    }


    if not args.dry_run:

        write_manifest(
            manifest_path,
            manifest,
        )


    # ========================================================
    # HEADER
    # ========================================================

    print("")
    print("=" * 72)
    print("CANONICAL POSSESSION PIPELINE")
    print("=" * 72)

    print(
        "Video:",
        video,
    )

    print(
        "Players:",
        players,
    )

    print(
        "Ball:",
        ball,
    )

    print(
        "Player model:",
        player_model,
    )

    print(
        "Run dir:",
        run_dir,
    )

    print(
        "Device:",
        args.device,
    )


    # ========================================================
    # 1. POSSESSION SEED
    # ========================================================

    command = [
        PYTHON,
        possession_seed_script,

        "--players",
        players,

        "--ball",
        ball,

        "--output",
        seed_frames,

        "--summary-output",
        seed_summary,

        "--intermediate-output",
        seed_intermediate,

        "--fps",
        str(
            args.fps
        ),
    ]

    run_stage(
        "1/7 POSSESSION SEED",
        command,
        [
            seed_frames,
            seed_summary,
            seed_intermediate,
        ],
        logs_dir
        /
        "01_possession_seed.log",
        resume=args.resume,
        dry_run=args.dry_run,
    )


    # ========================================================
    # 2. AIR INTERACTION
    # ========================================================

    command = [
        PYTHON,
        air_interaction_script,

        "--video",
        video,

        "--player-model",
        player_model,

        "--players",
        players,

        "--possession",
        seed_frames,

        "--output-interactions",
        air_interactions,

        "--output-v3-events",
        air_v3_events,

        "--output-v3-confirmed",
        air_v3_confirmed,

        "--output",
        air_events,

        "--output-confirmed",
        air_confirmed,

        "--output-contests",
        air_contests,

        "--device",
        args.device,
    ]


    air_outputs = [
        air_interactions,
        air_v3_events,
        air_v3_confirmed,
        air_events,
        air_confirmed,
        air_contests,
    ]


    if args.skip_qa:

        command.append(
            "--skip-qa"
        )

    else:

        command.extend(
            [
                "--output-video",
                air_qa_video,
            ]
        )

        air_outputs.append(
            air_qa_video
        )


    run_stage(
        "2/7 AIR INTERACTION",
        command,
        air_outputs,
        logs_dir
        /
        "02_air_interaction.log",
        resume=args.resume,
        dry_run=args.dry_run,
    )


    # ========================================================
    # 3. PHYSICAL EVENT / LAST TOUCH INTEGRATION
    # ========================================================

    command = [
        PYTHON,
        possession_v5_1_script,

        "--possession",
        seed_frames,

        "--events",
        air_events,

        "--output",
        v5_1_frames,

        "--summary-output",
        v5_1_summary,

        "--audit-output",
        v5_1_audit,
    ]


    run_stage(
        "3/7 PHYSICAL EVENT INTEGRATION",
        command,
        [
            v5_1_frames,
            v5_1_summary,
            v5_1_audit,
        ],
        logs_dir
        /
        "03_physical_event_integration.log",
        resume=args.resume,
        dry_run=args.dry_run,
    )


    # ========================================================
    # 4. CONTROLLER VALIDATION
    # ========================================================

    command = [
        PYTHON,
        controller_gate_script,

        "--video",
        video,

        "--player-model",
        player_model,

        "--players",
        players,

        "--possession",
        v5_1_frames,

        "--events",
        air_events,

        "--output",
        gate_frames,

        "--flagged-output",
        gate_flagged,

        "--device",
        args.device,
    ]


    gate_outputs = [
        gate_frames,
        gate_flagged,
    ]


    if args.skip_qa:

        command.append(
            "--skip-qa"
        )

    else:

        command.extend(
            [
                "--output-video",
                gate_qa_video,
            ]
        )

        gate_outputs.append(
            gate_qa_video
        )


    run_stage(
        "4/7 CONTROLLER VALIDATION",
        command,
        gate_outputs,
        logs_dir
        /
        "04_controller_validation.log",
        resume=args.resume,
        dry_run=args.dry_run,
    )


    # ========================================================
    # 5. POSSESSION REFINEMENT
    # ========================================================

    command = [
        PYTHON,
        refinement_script,

        "--possession",
        v5_1_frames,

        "--gate",
        gate_frames,

        "--events",
        air_events,

        "--intermediate-output",
        v5_2_frames,

        "--intermediate-summary-output",
        v5_2_summary,

        "--intermediate-audit-output",
        v5_2_audit,

        "--output",
        v5_3_frames,

        "--summary-output",
        v5_3_summary,

        "--audit-output",
        v5_3_audit,

        "--episodes-output",
        contest_episodes,
    ]


    run_stage(
        "5/7 POSSESSION REFINEMENT",
        command,
        [
            v5_2_frames,
            v5_2_summary,
            v5_2_audit,

            v5_3_frames,
            v5_3_summary,
            v5_3_audit,

            contest_episodes,
        ],
        logs_dir
        /
        "05_possession_refinement.log",
        resume=args.resume,
        dry_run=args.dry_run,
    )


    # ========================================================
    # 6. BALL MOTION
    # ========================================================

    command = [
        PYTHON,
        ball_motion_script,

        "--possession",
        v5_3_frames,

        "--gate",
        gate_frames,

        "--events",
        air_events,

        "--output-dir",
        data_dir,

        "--launch-min-speed",
        str(
            args.launch_min_speed
        ),

        "--reception-confirm-sec",
        str(
            args.reception_confirm_sec
        ),

        "--dribble-return-sec",
        str(
            args.dribble_return_sec
        ),
    ]


    run_stage(
        "6/7 BALL MOTION",
        command,
        [
            motion_frames,
            motion_episodes,
            motion_receptions,
            motion_dribble_gaps,
        ],
        logs_dir
        /
        "06_ball_motion.log",
        resume=args.resume,
        dry_run=args.dry_run,
    )


    # ========================================================
    # 7. FINAL POSSESSION
    # ========================================================

    command = [
        PYTHON,
        possession_v6_script,

        "--motion",
        motion_frames,

        "--output-dir",
        data_dir,
    ]


    run_stage(
        "7/7 FINAL POSSESSION",
        command,
        [
            v6_frames,
            v6_episodes,
        ],
        logs_dir
        /
        "07_final_possession.log",
        resume=args.resume,
        dry_run=args.dry_run,
    )


    # ========================================================
    # PRODUCTION ALIASES
    # ========================================================

    if not args.dry_run:

        shutil.copy2(
            v6_frames,
            final_frames,
        )

        shutil.copy2(
            v6_episodes,
            final_episodes,
        )


        manifest[
            "completed_at_utc"
        ] = datetime.now(
            timezone.utc
        ).isoformat()


        manifest[
            "status"
        ] = "completed"


        write_manifest(
            manifest_path,
            manifest,
        )


    # ========================================================
    # DONE
    # ========================================================

    print("")
    print("=" * 72)
    print("POSSESSION PIPELINE COMPLETE")
    print("=" * 72)

    print("")
    print(
        "Final frames:",
        final_frames,
    )

    print(
        "Final episodes:",
        final_episodes,
    )

    print(
        "Manifest:",
        manifest_path,
    )


if __name__ == "__main__":
    main()
