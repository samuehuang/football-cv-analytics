#!/usr/bin/env python3

from pathlib import Path
import csv
import importlib
import py_compile
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_file(relative_path):
    path = ROOT / relative_path
    check(path.is_file(), "Missing file: {}".format(relative_path))
    print("[PASS] file:", relative_path)


def check_csv(relative_path, required_columns):
    path = ROOT / relative_path

    check(path.is_file(), "Missing CSV: {}".format(relative_path))

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])

    missing = [
        column
        for column in required_columns
        if column not in header
    ]

    check(
        not missing,
        "{} missing columns: {}".format(
            relative_path,
            ", ".join(missing),
        ),
    )

    print(
        "[PASS] schema:",
        relative_path,
        "({} columns)".format(len(header)),
    )


def check_import(module_name):
    importlib.import_module(module_name)
    print("[PASS] import:", module_name)


def check_python_sources():
    files = [
        ROOT / "run_pipeline.py",
        ROOT / "run_analytics.py",
    ]

    files.extend(
        sorted((ROOT / "src").rglob("*.py"))
    )

    for path in files:
        py_compile.compile(
            str(path),
            doraise=True,
        )

    print(
        "[PASS] Python compilation: {} files".format(
            len(files)
        )
    )


def check_cli(script, expected_tokens):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / script),
            "--help",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    check(
        result.returncode == 0,
        "{} --help failed:\n{}".format(
            script,
            result.stdout,
        ),
    )

    for token in expected_tokens:
        check(
            token in result.stdout,
            "{} --help missing {}".format(
                script,
                token,
            ),
        )

    print("[PASS] CLI:", script, "--help")


def main():
    print("=" * 72)
    print("FOOTBALL CV ANALYTICS - SMOKE TEST")
    print("=" * 72)

    print()
    print("Repository structure")

    for relative_path in [
        "run_pipeline.py",
        "run_analytics.py",
        "configs/default.yaml",
        "requirements.txt",
        "README.md",
    ]:
        check_file(relative_path)

    print()
    print("Direct dependencies")

    for module_name in [
        "cv2",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "sports",
        "supervision",
        "torch",
        "ultralytics",
        "yaml",
    ]:
        check_import(module_name)

    print()
    print("Python source")

    check_python_sources()

    print()
    print("Command-line interfaces")

    check_cli(
        "run_pipeline.py",
        [
            "--video",
            "--run-name",
            "--config",
            "--device",
            "--resume",
            "--overwrite",
            "--dry-run",
        ],
    )

    check_cli(
        "run_analytics.py",
        [
            "--run-dir",
            "--tracking",
            "--overwrite",
            "--dry-run",
        ],
    )

    print()
    print("Sample-output schemas")

    check_csv(
        "examples/sample_outputs/tracking_sample.csv",
        [
            "frame",
            "time_sec",
            "track_id",
            "team",
            "pitch_x_final_m",
            "pitch_y_final_m",
            "speed_final_mps",
        ],
    )

    check_csv(
        "examples/sample_outputs/football_events.csv",
        [
            "event_id",
            "event_type",
            "start_frame",
            "end_frame",
            "team",
            "confidence",
            "source",
        ],
    )

    check_csv(
        "examples/sample_outputs/tactical_coverage_comparison.csv",
        [
            "mode",
            "complete_frames",
            "total_frames",
            "coverage_pct",
        ],
    )

    check_csv(
        "examples/sample_outputs/team_tactical_metrics_summary_identity_aware.csv",
        [
            "team",
            "valid_frames",
            "centroid_x_m_median",
            "centroid_y_m_median",
            "team_length_m_median",
            "team_width_m_median",
            "compactness_m_median",
            "shape_area_m2_median",
        ],
    )

    print()
    print("=" * 72)
    print("SMOKE TEST: PASS")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 72)
        print("SMOKE TEST: FAIL")
        print("=" * 72)
        print(exc)
        sys.exit(1)
