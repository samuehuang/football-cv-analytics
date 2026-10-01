#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


# ============================================================
# Canonical implementation
# ============================================================

CANONICAL_SCRIPT = (
    Path(__file__)
    .resolve()
    .with_name(
        "inter_team_spatial_metrics.py"
    )
)


# ============================================================
# Identity-aware paths
# ============================================================

INPUT_CSV = (
    "outputs/"
    "tracking_data_tactical_identity_aware.csv"
)

FRAME_OUTPUT = (
    "outputs/"
    "inter_team_spatial_metrics_frame_identity_aware.csv"
)

SUMMARY_OUTPUT = (
    "outputs/"
    "inter_team_spatial_metrics_summary_identity_aware.csv"
)


# ============================================================
# Load canonical source
# ============================================================

if not CANONICAL_SCRIPT.is_file():
    raise FileNotFoundError(
        "Canonical inter-team spatial metrics "
        f"script not found: {CANONICAL_SCRIPT}"
    )


source = CANONICAL_SCRIPT.read_text(
    encoding="utf-8"
)


# ============================================================
# Exact path replacements only
#
# The metric implementation itself remains canonical.
# Fail loudly if the canonical source changes so that we do
# not silently run a different algorithm.
# ============================================================

replacements = [
    (
        'INPUT_CSV = "outputs/tracking_data_tactical.csv"',
        (
            'INPUT_CSV = '
            f'"{INPUT_CSV}"'
        ),
    ),
    (
        (
            'FRAME_OUTPUT = '
            '"outputs/inter_team_spatial_metrics_frame.csv"'
        ),
        (
            'FRAME_OUTPUT = '
            f'"{FRAME_OUTPUT}"'
        ),
    ),
    (
        (
            'SUMMARY_OUTPUT = '
            '"outputs/inter_team_spatial_metrics_summary.csv"'
        ),
        (
            'SUMMARY_OUTPUT = '
            f'"{SUMMARY_OUTPUT}"'
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


# ============================================================
# Display identity-aware execution context
# ============================================================

print()
print("=" * 70)
print(
    "IDENTITY-AWARE INTER-TEAM SPATIAL METRICS"
)
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
    "Frame output:",
    FRAME_OUTPUT,
)

print(
    "Summary output:",
    SUMMARY_OUTPUT,
)

print()


# ============================================================
# Execute the canonical implementation
# ============================================================

code = compile(
    source,
    str(CANONICAL_SCRIPT),
    "exec",
)

namespace = {
    "__name__": "__main__",
    "__file__": str(
        CANONICAL_SCRIPT
    ),
}

exec(
    code,
    namespace,
    namespace,
)
