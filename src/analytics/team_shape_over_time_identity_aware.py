#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


CANONICAL_SCRIPT = (
    Path(__file__)
    .resolve()
    .with_name("team_shape_over_time_v2.py")
)


INPUT_CSV = (
    "outputs/"
    "team_tactical_metrics_frame_identity_aware.csv"
)

LENGTH_OUTPUT = (
    "outputs/"
    "team_length_over_time_identity_aware.png"
)

WIDTH_OUTPUT = (
    "outputs/"
    "team_width_over_time_identity_aware.png"
)

COMPACT_OUTPUT = (
    "outputs/"
    "team_compactness_over_time_identity_aware.png"
)


if not CANONICAL_SCRIPT.is_file():
    raise FileNotFoundError(
        f"Canonical team-shape script not found: "
        f"{CANONICAL_SCRIPT}"
    )


source = CANONICAL_SCRIPT.read_text(
    encoding="utf-8"
)


replacements = [
    (
        'INPUT_CSV = "outputs/team_tactical_metrics_frame_v2.csv"',
        f'INPUT_CSV = "{INPUT_CSV}"',
    ),
    (
        'LENGTH_OUTPUT = "outputs/team_length_over_time_v2.png"',
        f'LENGTH_OUTPUT = "{LENGTH_OUTPUT}"',
    ),
    (
        'WIDTH_OUTPUT = "outputs/team_width_over_time_v2.png"',
        f'WIDTH_OUTPUT = "{WIDTH_OUTPUT}"',
    ),
    (
        'COMPACT_OUTPUT = "outputs/team_compactness_over_time_v2.png"',
        f'COMPACT_OUTPUT = "{COMPACT_OUTPUT}"',
    ),
    (
        'print("CLEAN TEAM SHAPE OVER TIME")',
        'print("IDENTITY-AWARE TEAM SHAPE OVER TIME")',
    ),
    (
        'title="Team Length Over Time — Clean 10v10 Frames"',
        (
            'title="Team Length Over Time — '
            'Identity-Aware 10v10 Frames"'
        ),
    ),
    (
        'title="Team Width Over Time — Clean 10v10 Frames"',
        (
            'title="Team Width Over Time — '
            'Identity-Aware 10v10 Frames"'
        ),
    ),
    (
        'title="Team Compactness Over Time — Clean 10v10 Frames"',
        (
            'title="Team Compactness Over Time — '
            'Identity-Aware 10v10 Frames"'
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
print("IDENTITY-AWARE TEAM SHAPE OVER TIME")
print("=" * 70)

print(
    "Canonical implementation:",
    CANONICAL_SCRIPT,
)

print(
    "Input:",
    INPUT_CSV,
)

print(
    "Length output:",
    LENGTH_OUTPUT,
)

print(
    "Width output:",
    WIDTH_OUTPUT,
)

print(
    "Compactness output:",
    COMPACT_OUTPUT,
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
