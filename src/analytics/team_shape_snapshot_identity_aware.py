#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


CANONICAL_SCRIPT = (
    Path(__file__)
    .resolve()
    .with_name("team_shape_snapshot.py")
)


TRACKING_CSV = (
    "outputs/"
    "tracking_data_tactical_identity_aware.csv"
)

METRICS_CSV = (
    "outputs/"
    "team_tactical_metrics_frame_identity_aware.csv"
)

OUTPUT_PATH = (
    "outputs/"
    "team_shape_snapshot_identity_aware.png"
)


if not CANONICAL_SCRIPT.is_file():
    raise FileNotFoundError(
        f"Canonical team-shape snapshot script "
        f"not found: {CANONICAL_SCRIPT}"
    )


source = CANONICAL_SCRIPT.read_text(
    encoding="utf-8"
)


replacements = [
    (
        'TRACKING_CSV = "outputs/tracking_data_tactical.csv"',
        f'TRACKING_CSV = "{TRACKING_CSV}"',
    ),
    (
        'METRICS_CSV = "outputs/team_tactical_metrics_frame_v2.csv"',
        f'METRICS_CSV = "{METRICS_CSV}"',
    ),
    (
        'OUTPUT_PATH = "outputs/team_shape_snapshot.png"',
        f'OUTPUT_PATH = "{OUTPUT_PATH}"',
    ),
    (
        'print("REPRESENTATIVE TACTICAL FRAME")',
        (
            'print('
            '"IDENTITY-AWARE REPRESENTATIVE TACTICAL FRAME"'
            ')'
        ),
    ),
    (
        '"Representative 10v10 Team Shape\\n"',
        (
            '"Representative 10v10 Team Shape '
            '— Identity-Aware\\n"'
        ),
    ),
]


for old, new in replacements:
    count = source.count(old)

    if count != 1:
        raise RuntimeError(
            "Canonical source contract changed.\n"
            f"Expected exactly one occurrence of:\n"
            f"{old}\n"
            f"Found: {count}"
        )

    source = source.replace(
        old,
        new,
        1,
    )


print()
print("=" * 70)
print("IDENTITY-AWARE TEAM SHAPE SNAPSHOT")
print("=" * 70)

print(
    "Canonical implementation:",
    CANONICAL_SCRIPT,
)

print(
    "Tracking input:",
    TRACKING_CSV,
)

print(
    "Metrics input:",
    METRICS_CSV,
)

print(
    "Output:",
    OUTPUT_PATH,
)

print()


code = compile(
    source,
    str(CANONICAL_SCRIPT),
    "exec",
)

namespace = {
    "__name__": "__main__",
    "__file__": str(CANONICAL_SCRIPT),
}

exec(
    code,
    namespace,
    namespace,
)
