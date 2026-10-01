import os
import pandas as pd
import numpy as np


TRACKING_CSV = "outputs/tracking_data_clean_v2.csv"
METRICS_CSV = "outputs/player_metrics.csv"

OUTPUT_CSV = "outputs/tracking_data_tactical.csv"
FRAME_QUALITY_CSV = "outputs/tactical_frame_quality.csv"

MIN_TRACKED_TIME = 20.0

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Load
# ============================================================

tracking = pd.read_csv(TRACKING_CSV)
metrics = pd.read_csv(METRICS_CSV)


tracking = tracking[
    tracking["team"].isin(["A", "B"])
].copy()


# ============================================================
# 1. Select long-term core tracks
# ============================================================

core = metrics[
    metrics["tracked_time_sec"] >= MIN_TRACKED_TIME
][
    [
        "track_id",
        "team",
        "tracked_time_sec"
    ]
].copy()


print("")
print("========================================")
print("CORE TRACKS")
print("========================================")
print("")

print(
    core.sort_values(
        ["team", "track_id"]
    ).to_string(index=False)
)


print("")

print(
    "Team A:",
    (core["team"] == "A").sum()
)

print(
    "Team B:",
    (core["team"] == "B").sum()
)


# ============================================================
# Dominant / stable team mapping
# ============================================================

stable_team = dict(
    zip(
        core["track_id"],
        core["team"]
    )
)


core_ids = set(
    core["track_id"]
)


# ============================================================
# 2. Only core IDs
# ============================================================

df = tracking[
    tracking["track_id"].isin(core_ids)
].copy()


# ============================================================
# 3. Check row-level team consistency
# ============================================================

df["stable_team"] = (
    df["track_id"]
    .map(stable_team)
)


df["team_consistent"] = (
    df["team"]
    ==
    df["stable_team"]
)


print("")
print("========================================")
print("TEAM LABEL INCONSISTENCY")
print("========================================")
print("")


inconsistent = df[
    ~df["team_consistent"]
].copy()


if len(inconsistent) == 0:

    print(
        "No inconsistent team labels."
    )

else:

    summary = (
        inconsistent
        .groupby(
            [
                "track_id",
                "stable_team",
                "team"
            ]
        )
        .agg(
            rows=("frame", "size"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            first_time=("time_sec", "min"),
            last_time=("time_sec", "max")
        )
        .reset_index()
    )


    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# IMPORTANT:
#
# Do NOT simply relabel inconsistent rows.
#
# An inconsistency may indicate ID switch.
# Therefore we REMOVE those observations.
# ============================================================

df = df[
    df["team_consistent"]
].copy()


# From now on use stable team
df["team"] = df["stable_team"]


# ============================================================
# 4. Frame composition quality
# ============================================================

quality_rows = []


for frame_id, frame_df in df.groupby("frame"):

    time_sec = float(
        frame_df["time_sec"].iloc[0]
    )


    a = frame_df[
        frame_df["team"] == "A"
    ]

    b = frame_df[
        frame_df["team"] == "B"
    ]


    a_ids = sorted(
        a["track_id"]
        .astype(int)
        .unique()
    )

    b_ids = sorted(
        b["track_id"]
        .astype(int)
        .unique()
    )


    a_count = len(a_ids)
    b_count = len(b_ids)


    complete = (
        a_count == 10
        and
        b_count == 10
    )


    quality_rows.append(
        {
            "frame":
                int(frame_id),

            "time_sec":
                time_sec,

            "team_A_count":
                a_count,

            "team_B_count":
                b_count,

            "team_A_ids":
                ",".join(
                    map(str, a_ids)
                ),

            "team_B_ids":
                ",".join(
                    map(str, b_ids)
                ),

            "complete_20_players":
                complete
        }
    )


quality = pd.DataFrame(
    quality_rows
)


quality.to_csv(
    FRAME_QUALITY_CSV,
    index=False
)


# ============================================================
# 5. Keep only complete frames
# ============================================================

valid_frames = set(
    quality[
        quality[
            "complete_20_players"
        ]
    ]["frame"]
)


tactical = df[
    df["frame"].isin(valid_frames)
].copy()


# ============================================================
# Save
# ============================================================

tactical.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Report
# ============================================================

total_frames = (
    tracking["frame"]
    .nunique()
)

complete_frames = (
    len(valid_frames)
)


print("")
print("========================================")
print("TACTICAL DATA QUALITY")
print("========================================")
print("")

print(
    "Original frames:",
    total_frames
)

print(
    "Complete tactical frames:",
    complete_frames
)

print(
    "Coverage:",
    f"{complete_frames / total_frames * 100:.1f}%"
)


if complete_frames > 0:

    valid_times = tactical[
        "time_sec"
    ]

    print(
        "First valid time:",
        f"{valid_times.min():.2f}s"
    )

    print(
        "Last valid time:",
        f"{valid_times.max():.2f}s"
    )


print("")
print(
    "Rows:",
    len(tactical)
)

print(
    "Expected rows:",
    complete_frames * 20
)


print("")
print(
    "Saved:",
    OUTPUT_CSV
)

print(
    "Frame quality:",
    FRAME_QUALITY_CSV
)
