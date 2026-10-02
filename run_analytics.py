#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


ANALYTICS_STAGES = [
    (
        "identity_qa",
        ROOT / "src" / "analytics" / "identity_qa.py",
    ),
    (
        "player_metrics",
        ROOT / "src" / "analytics" / "player_metrics.py",
    ),
    (
        "build_tactical_clean_dataset",
        ROOT / "src" / "analytics" / "build_tactical_clean_dataset.py",
    ),
    (
        "build_tactical_identity_aware_dataset",
        ROOT
        / "src"
        / "analytics"
        / "build_tactical_identity_aware_dataset.py",
    ),
    (
        "team_tactical_metrics",
        ROOT / "src" / "analytics" / "team_tactical_metrics_v2.py",
    ),
    (
        "team_tactical_metrics_identity_aware",
        ROOT
        / "src"
        / "analytics"
        / "team_tactical_metrics_identity_aware.py",
    ),
    (
        "inter_team_spatial_metrics",
        ROOT / "src" / "analytics" / "inter_team_spatial_metrics.py",
    ),
    (
        "inter_team_spatial_metrics_identity_aware",
        ROOT
        / "src"
        / "analytics"
        / "inter_team_spatial_metrics_identity_aware.py",
    ),
    (
        "player_spatial_analysis",
        ROOT / "src" / "analytics" / "player_spatial_analysis.py",
    ),
    (
        "team_composition_check",
        ROOT / "src" / "analytics" / "team_composition_check.py",
    ),
    (
        "team_shape_over_time",
        ROOT / "src" / "analytics" / "team_shape_over_time_v2.py",
    ),
    (
        "team_shape_over_time_identity_aware",
        ROOT
        / "src"
        / "analytics"
        / "team_shape_over_time_identity_aware.py",
    ),
    (
        "team_shape_snapshot",
        ROOT / "src" / "analytics" / "team_shape_snapshot.py",
    ),
    (
        "team_shape_snapshot_identity_aware",
        ROOT
        / "src"
        / "analytics"
        / "team_shape_snapshot_identity_aware.py",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the optional football tactical analytics pipeline "
            "on outputs from the canonical tracking pipeline."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Canonical pipeline run directory, "
            "for example: runs/my_match"
        ),
    )

    parser.add_argument(
        "--tracking",
        default=None,
        help=(
            "Optional override for tracking_data_clean_v2.csv. "
            "Default: <run-dir>/data/tracking_data_clean_v2.csv"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing analytics directory.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned stages without executing them.",
    )

    return parser.parse_args()


def print_stage(
    name: str,
    command: list[str],
) -> None:
    print()
    print("=" * 70)
    print(f"STAGE: {name}")
    print("=" * 70)
    print("COMMAND:")
    print(" ".join(command))


def main() -> None:
    args = parse_args()

    run_dir = (
        Path(args.run_dir)
        .expanduser()
        .resolve()
    )

    if args.tracking:
        tracking_csv = (
            Path(args.tracking)
            .expanduser()
            .resolve()
        )
    else:
        tracking_csv = (
            run_dir
            / "data"
            / "tracking_data_clean_v2.csv"
        )

    analytics_dir = (
        run_dir
        / "analytics"
    )

    outputs_dir = (
        analytics_dir
        / "outputs"
    )

    print(
        f"RUN DIR: {run_dir}"
    )

    print(
        f"TRACKING: {tracking_csv}"
    )

    print(
        f"ANALYTICS DIR: {analytics_dir}"
    )

    if not tracking_csv.exists():
        raise FileNotFoundError(
            f"Tracking input not found: "
            f"{tracking_csv}"
        )

    for _, script in ANALYTICS_STAGES:
        if not script.exists():
            raise FileNotFoundError(
                f"Analytics script not found: "
                f"{script}"
            )

    if analytics_dir.exists():
        has_content = any(
            analytics_dir.iterdir()
        )

        if (
            has_content
            and not args.overwrite
        ):
            raise FileExistsError(
                f"Analytics directory already exists: "
                f"{analytics_dir}\n"
                f"Use --overwrite to replace it."
            )

        if args.overwrite:
            if args.dry_run:
                print(
                    f"[DRY RUN] Would remove existing: "
                    f"{analytics_dir}"
                )
            else:
                shutil.rmtree(
                    analytics_dir
                )

    if args.dry_run:
        print()
        print("INPUT STAGING")

        print(
            f"{tracking_csv} -> "
            f"{outputs_dir / 'tracking_data_clean_v2.csv'}"
        )

        for name, script in ANALYTICS_STAGES:
            command = [
                sys.executable,
                str(script),
            ]

            print_stage(
                name,
                command,
            )

        print()
        print("=" * 70)
        print(
            "ANALYTICS DRY RUN COMPLETE"
        )
        print("=" * 70)

        return

    outputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    staged_tracking = (
        outputs_dir
        / "tracking_data_clean_v2.csv"
    )

    shutil.copy2(
        tracking_csv,
        staged_tracking,
    )

    print()

    print(
        f"Staged tracking input: "
        f"{staged_tracking}"
    )

    env = os.environ.copy()

    env.setdefault(
        "MPLBACKEND",
        "Agg",
    )

    env[
        "PYTHONUNBUFFERED"
    ] = "1"

    for name, script in ANALYTICS_STAGES:
        command = [
            sys.executable,
            str(script),
        ]

        print_stage(
            name,
            command,
        )

        subprocess.run(
            command,
            cwd=analytics_dir,
            check=True,
            env=env,
        )

    print()
    print("=" * 70)
    print(
        "ANALYTICS PIPELINE COMPLETE"
    )
    print("=" * 70)

    print(
        f"Analytics run: "
        f"{analytics_dir}"
    )

    print(
        f"Outputs:       "
        f"{outputs_dir}"
    )


if __name__ == "__main__":
    main()