import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = "outputs/tracking_data_clean_v2.csv"

OUTPUT_CSV = "outputs/team_composition_by_frame.csv"
OUTPUT_PLOT = "outputs/team_player_count_over_time.png"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = df[
    df["team"].isin(["A", "B"])
].copy()


# ============================================================
# Frame-by-frame composition
# ============================================================

rows = []


for frame_id, frame_df in df.groupby("frame"):

    time_sec = float(
        frame_df["time_sec"].iloc[0]
    )


    for team_name in ["A", "B"]:

        team = frame_df[
            frame_df["team"] == team_name
        ]


        ids = sorted(
            team["track_id"]
            .astype(int)
            .unique()
            .tolist()
        )


        rows.append(
            {
                "frame":
                    int(frame_id),

                "time_sec":
                    time_sec,

                "team":
                    team_name,

                "player_count":
                    len(ids),

                "track_ids":
                    ",".join(
                        map(str, ids)
                    )
            }
        )


composition = pd.DataFrame(rows)

composition.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Count distribution
# ============================================================

print("")
print("========================================")
print("TEAM COMPOSITION CHECK")
print("========================================")


for team_name in ["A", "B"]:

    team = composition[
        composition["team"] == team_name
    ]


    print("")
    print(
        f"Team {team_name} player-count distribution:"
    )


    print(
        team["player_count"]
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# Detect composition changes
# ============================================================

print("")
print("========================================")
print("TRACK-ID COMPOSITION CHANGES")
print("========================================")


for team_name in ["A", "B"]:

    team = (
        composition[
            composition["team"]
            == team_name
        ]
        .sort_values("frame")
        .copy()
    )


    team["previous_ids"] = (
        team["track_ids"].shift(1)
    )


    changes = team[
        team["track_ids"]
        != team["previous_ids"]
    ].copy()


    # 第一幀不是實際 change
    changes = changes[
        changes["frame"]
        != team["frame"].min()
    ]


    print("")
    print(
        f"Team {team_name}"
    )

    print(
        "Number of composition changes:",
        len(changes)
    )


    if len(changes) > 0:

        print("")
        print(
            changes[
                [
                    "frame",
                    "time_sec",
                    "player_count",
                    "previous_ids",
                    "track_ids"
                ]
            ]
            .head(50)
            .to_string(index=False)
        )


# ============================================================
# Identify count-change moments
# ============================================================

print("")
print("========================================")
print("PLAYER COUNT CHANGES")
print("========================================")


for team_name in ["A", "B"]:

    team = (
        composition[
            composition["team"]
            == team_name
        ]
        .sort_values("frame")
        .copy()
    )


    team["previous_count"] = (
        team["player_count"]
        .shift(1)
    )


    count_changes = team[
        team["player_count"]
        != team["previous_count"]
    ]


    print("")
    print(
        f"Team {team_name}"
    )


    print(
        count_changes[
            [
                "frame",
                "time_sec",
                "previous_count",
                "player_count",
                "track_ids"
            ]
        ]
        .head(50)
        .to_string(index=False)
    )


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 5)
)


for team_name in ["A", "B"]:

    team = composition[
        composition["team"]
        == team_name
    ]


    ax.plot(
        team["time_sec"],
        team["player_count"],
        linewidth=2,
        label=f"Team {team_name}"
    )


ax.axhline(
    10,
    linestyle="--",
    linewidth=1,
    alpha=0.5
)


ax.set_xlabel(
    "Time (seconds)"
)

ax.set_ylabel(
    "Tracked Outfield Players"
)

ax.set_title(
    "Tracked Players Per Team Over Time"
)

ax.set_ylim(
    5,
    12
)

ax.grid(
    alpha=0.2
)

ax.legend()

plt.tight_layout()


plt.savefig(
    OUTPUT_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("")
print("========================================")
print("DONE")
print("========================================")

print(
    "CSV:",
    OUTPUT_CSV
)

print(
    "Plot:",
    OUTPUT_PLOT
)
