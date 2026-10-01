import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc


CSV_PATH = "outputs/tracking_data_v4.csv"
OUTPUT_PATH = "outputs/player_trajectories_v4.png"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Rows:", len(df))
print("Unique Track IDs:", df["track_id"].nunique())


# ============================================================
# Remove GK for now
# ============================================================

players = df[
    df["team"].isin(["A", "B"])
].copy()


# ============================================================
# Find longest tracks
#
# 不要一次畫所有 ID，不然畫面會非常亂
# 先取存在 frame 數最多的球員
# ============================================================

track_lengths = (
    players
    .groupby("track_id")
    .size()
    .sort_values(ascending=False)
)

print("")
print("Longest tracks:")
print(track_lengths.head(15))


# 取前 10 個最長 trajectory
TOP_N = 10

selected_ids = (
    track_lengths
    .head(TOP_N)
    .index
    .tolist()
)

print("")
print("Selected Track IDs:")
print(selected_ids)


# ============================================================
# Football pitch
# 120m x 70m
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 8)
)

ax.set_xlim(0, 120)
ax.set_ylim(0, 70)

ax.set_aspect("equal")


# ============================================================
# Pitch boundary
# ============================================================

pitch = Rectangle(
    (0, 0),
    120,
    70,
    fill=False,
    linewidth=2
)

ax.add_patch(pitch)


# ============================================================
# Halfway line
# ============================================================

ax.plot(
    [60, 60],
    [0, 70],
    linewidth=1.5
)


# ============================================================
# Centre circle
# ============================================================

center_circle = Circle(
    (60, 35),
    9.15,
    fill=False,
    linewidth=1.5
)

ax.add_patch(center_circle)

ax.scatter(
    [60],
    [35],
    s=15
)


# ============================================================
# Penalty areas
# ============================================================

left_penalty = Rectangle(
    (0, 13.85),
    16.5,
    40.3,
    fill=False,
    linewidth=1.5
)

right_penalty = Rectangle(
    (103.5, 13.85),
    16.5,
    40.3,
    fill=False,
    linewidth=1.5
)

ax.add_patch(left_penalty)
ax.add_patch(right_penalty)


# ============================================================
# Goal areas
# ============================================================

left_goal_area = Rectangle(
    (0, 24),
    5.5,
    22,
    fill=False,
    linewidth=1.5
)

right_goal_area = Rectangle(
    (114.5, 24),
    5.5,
    22,
    fill=False,
    linewidth=1.5
)

ax.add_patch(left_goal_area)
ax.add_patch(right_goal_area)


# ============================================================
# Penalty spots
# ============================================================

ax.scatter(
    [11, 109],
    [35, 35],
    s=12
)


# ============================================================
# Draw trajectories
# ============================================================

for track_id in selected_ids:

    track = players[
        players["track_id"] == track_id
    ].sort_values(
        "frame"
    )

    if len(track) < 2:
        continue

    team = track["team"].mode()[0]

    x = track["pitch_x_m"].values
    y = track["pitch_y_m"].values


    # Trajectory
    ax.plot(
        x,
        y,
        linewidth=1.5,
        alpha=0.75,
        label=f"{team} #{track_id}"
    )


    # Start
    ax.scatter(
        x[0],
        y[0],
        marker="o",
        s=35
    )


    # End
    ax.scatter(
        x[-1],
        y[-1],
        marker="x",
        s=50
    )


    # Track ID label
    ax.text(
        x[-1],
        y[-1],
        f" #{track_id}",
        fontsize=8
    )


# ============================================================
# Plot settings
# ============================================================

ax.set_title(
    "Football Player Trajectories"
)

ax.set_xlabel(
    "Pitch X (m)"
)

ax.set_ylabel(
    "Pitch Y (m)"
)


ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=5,
    fontsize=8
)


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("")
print("Done!")
print("Saved:", OUTPUT_PATH)
