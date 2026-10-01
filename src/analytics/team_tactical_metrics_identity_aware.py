#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


# ============================================================
# Identity-aware Team Tactical Metrics
#
# IMPORTANT:
# This file intentionally reuses the exact implementation from
# team_tactical_metrics_v2.py.
#
# Only the input/output paths are changed.
#
# This guarantees:
#
# strict dataset
#     and
# identity-aware dataset
#
# are evaluated using exactly the same tactical metrics logic.
# ============================================================


HERE = Path(__file__).resolve().parent

SOURCE_SCRIPT = (
    HERE
    / "team_tactical_metrics_v2.py"
)


# ============================================================
# Expected source paths
# ============================================================

STRICT_INPUT = (
    'INPUT_CSV = '
    '"outputs/tracking_data_tactical.csv"'
)

STRICT_FRAME_OUTPUT = (
    'FRAME_OUTPUT = '
    '"outputs/team_tactical_metrics_frame_v2.csv"'
)

STRICT_SUMMARY_OUTPUT = (
    'SUMMARY_OUTPUT = '
    '"outputs/team_tactical_metrics_summary_v2.csv"'
)


# ============================================================
# Identity-aware paths
# ============================================================

IDENTITY_INPUT = (
    'INPUT_CSV = '
    '"outputs/'
    'tracking_data_tactical_identity_aware.csv"'
)

IDENTITY_FRAME_OUTPUT = (
    'FRAME_OUTPUT = '
    '"outputs/'
    'team_tactical_metrics_frame_identity_aware.csv"'
)

IDENTITY_SUMMARY_OUTPUT = (
    'SUMMARY_OUTPUT = '
    '"outputs/'
    'team_tactical_metrics_summary_identity_aware.csv"'
)


def replace_exactly_once(
    source: str,
    old: str,
    new: str,
    label: str,
) -> str:
    """
    Replace one expected line exactly once.

    Fail loudly if the canonical source has changed.
    This prevents silently running an invalid comparison.
    """

    count = source.count(old)

    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {label} "
            f"in {SOURCE_SCRIPT}, "
            f"found {count}.\n"
            f"The canonical metrics script may have changed."
        )

    return source.replace(
        old,
        new,
        1,
    )


def main() -> None:
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Canonical tactical metrics script "
            f"not found: {SOURCE_SCRIPT}"
        )

    source = SOURCE_SCRIPT.read_text(
        encoding="utf-8"
    )

    # ========================================================
    # Replace only file paths.
    #
    # No metric formula, threshold, percentile,
    # filtering rule, or aggregation logic is changed.
    # ========================================================

    source = replace_exactly_once(
        source=source,
        old=STRICT_INPUT,
        new=IDENTITY_INPUT,
        label="INPUT_CSV assignment",
    )

    source = replace_exactly_once(
        source=source,
        old=STRICT_FRAME_OUTPUT,
        new=IDENTITY_FRAME_OUTPUT,
        label="FRAME_OUTPUT assignment",
    )

    source = replace_exactly_once(
        source=source,
        old=STRICT_SUMMARY_OUTPUT,
        new=IDENTITY_SUMMARY_OUTPUT,
        label="SUMMARY_OUTPUT assignment",
    )

    print()
    print("=" * 70)
    print(
        "IDENTITY-AWARE TEAM TACTICAL METRICS"
    )
    print("=" * 70)
    print(
        "Canonical implementation:",
        SOURCE_SCRIPT,
    )
    print(
        "Input:",
        "outputs/"
        "tracking_data_tactical_identity_aware.csv",
    )
    print(
        "Frame output:",
        "outputs/"
        "team_tactical_metrics_frame_identity_aware.csv",
    )
    print(
        "Summary output:",
        "outputs/"
        "team_tactical_metrics_summary_identity_aware.csv",
    )
    print()

    # ========================================================
    # Execute the canonical implementation.
    #
    # The compiled source is identical to
    # team_tactical_metrics_v2.py except for the three paths
    # replaced above.
    # ========================================================

    namespace = {
        "__name__": "__main__",
        "__file__": str(SOURCE_SCRIPT),
        "__package__": None,
    }

    compiled = compile(
        source,
        str(SOURCE_SCRIPT),
        "exec",
    )

    exec(
        compiled,
        namespace,
        namespace,
    )


if __name__ == "__main__":
    main()
