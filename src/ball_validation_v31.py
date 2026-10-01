import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

INPUT_CSV = "outputs/ball_tracking_v3.csv"

OUTPUT_CSV = "outputs/ball_tracking_v31_trusted.csv"

OUTPUT_VIDEO = "outputs/ball_tracking_v31_validation.mp4"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

FPS = 25.0


# ------------------------------------------------------------
# Confidence tiers
#
# >= 0.35
#   high-confidence observation
#
# 0.20 ~ 0.35
#   only keep when bidirectional temporal evidence supports it
#
# < 0.20
#   do NOT trust as a direct observation
#
# The 0.14 false positive you showed will therefore be rejected.
# ------------------------------------------------------------

HIGH_CONF = 0.35
MIN_DIRECT_CONF = 0.20


# ------------------------------------------------------------
# Medium-confidence validation
# ------------------------------------------------------------

# Look for high-confidence anchors within +/- 4 frames.
SUPPORT_WINDOW = 4

# Medium-confidence detection must stay close to the line
# implied by previous and next trustworthy anchors.
MAX_MEDIUM_RESIDUAL_M = 1.50


# ------------------------------------------------------------
# Short-gap interpolation
# ------------------------------------------------------------

MAX_INTERPOLATION_GAP = 3

# Do not interpolate between endpoints whose implied motion
# is already extremely fast.
MAX_INTERPOLATION_SPEED = 35.0


# ------------------------------------------------------------
# Ball physical / projection sanity
#
# This is NOT treated as a true football speed limit.
# V3 already used 60 m/s to eliminate obvious teleports.
# ------------------------------------------------------------

HIGH_SPEED_FLAG = 40.0


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_CSV)

df = (
    df
    .sort_values("frame")
    .reset_index(drop=True)
)


# ============================================================
# Normalize columns
# ============================================================

df["observed"] = (
    df["detected"].fillna(False).astype(bool)
    &
    df["projected"].fillna(False).astype(bool)
    &
    df["pitch_x_m"].notna()
    &
    df["pitch_y_m"].notna()
)


df["confidence"] = pd.to_numeric(
    df["confidence"],
    errors="coerce"
)


df["segment_id"] = pd.to_numeric(
    df["segment_id"],
    errors="coerce"
)


# ============================================================
# Output / validation columns
# ============================================================

df["trusted_observed"] = False

df["trusted_ball"] = False

df["interpolated"] = False

df["validation_reason"] = "missing"

df["temporal_residual_m"] = np.nan


# Clean coordinates
df["pitch_x_trusted_m"] = np.nan
df["pitch_y_trusted_m"] = np.nan

df["image_x_trusted"] = np.nan
df["image_y_trusted"] = np.nan


# ============================================================
# High-confidence anchors
# ============================================================

anchor_mask = (
    df["observed"]
    &
    (
        df["confidence"]
        >= HIGH_CONF
    )
)


# ============================================================
# Helper:
# Find previous / next high-confidence anchor
# in same segment.
# ============================================================

def find_anchor_before(
    row_index,
    segment_id,
):

    start = max(
        0,
        row_index - SUPPORT_WINDOW
    )


    for j in range(
        row_index - 1,
        start - 1,
        -1
    ):

        if not anchor_mask.iloc[j]:
            continue


        other_segment = df.loc[
            j,
            "segment_id"
        ]


        if (
            pd.isna(segment_id)
            or
            pd.isna(other_segment)
        ):

            continue


        if (
            other_segment
            ==
            segment_id
        ):

            return j


    return None


def find_anchor_after(
    row_index,
    segment_id,
):

    end = min(
        len(df),
        row_index + SUPPORT_WINDOW + 1
    )


    for j in range(
        row_index + 1,
        end
    ):

        if not anchor_mask.iloc[j]:
            continue


        other_segment = df.loc[
            j,
            "segment_id"
        ]


        if (
            pd.isna(segment_id)
            or
            pd.isna(other_segment)
        ):

            continue


        if (
            other_segment
            ==
            segment_id
        ):

            return j


    return None


# ============================================================
# PASS 1
# Validate direct observations
# ============================================================

medium_supported_count = 0

medium_rejected_count = 0

very_low_rejected_count = 0

high_conf_count = 0


for i in range(len(df)):

    if not bool(
        df.loc[
            i,
            "observed"
        ]
    ):

        df.loc[
            i,
            "validation_reason"
        ] = "missing"

        continue


    confidence = float(
        df.loc[
            i,
            "confidence"
        ]
    )


    segment_id = df.loc[
        i,
        "segment_id"
    ]


    # ========================================================
    # 1. High-confidence observation
    # ========================================================

    if confidence >= HIGH_CONF:

        df.loc[
            i,
            "trusted_observed"
        ] = True

        df.loc[
            i,
            "validation_reason"
        ] = "high_conf"

        high_conf_count += 1

        continue


    # ========================================================
    # 2. Very low confidence
    #
    # Hard reject as a DIRECT observation.
    #
    # It may later be replaced by interpolation if trustworthy
    # positions exist on both sides.
    # ========================================================

    if confidence < MIN_DIRECT_CONF:

        df.loc[
            i,
            "validation_reason"
        ] = "rejected_very_low_conf"

        very_low_rejected_count += 1

        continue


    # ========================================================
    # 3. Medium confidence:
    #    require evidence from BOTH temporal directions
    # ========================================================

    previous_anchor = find_anchor_before(
        i,
        segment_id
    )


    next_anchor = find_anchor_after(
        i,
        segment_id
    )


    if (
        previous_anchor is None
        or
        next_anchor is None
    ):

        df.loc[
            i,
            "validation_reason"
        ] = "rejected_no_bidirectional_support"

        medium_rejected_count += 1

        continue


    # ========================================================
    # Linear prediction from trustworthy endpoints
    # ========================================================

    frame_prev = float(
        df.loc[
            previous_anchor,
            "frame"
        ]
    )


    frame_now = float(
        df.loc[
            i,
            "frame"
        ]
    )


    frame_next = float(
        df.loc[
            next_anchor,
            "frame"
        ]
    )


    total_frame_distance = (
        frame_next
        -
        frame_prev
    )


    if total_frame_distance <= 0:

        df.loc[
            i,
            "validation_reason"
        ] = "rejected_invalid_temporal_geometry"

        medium_rejected_count += 1

        continue


    alpha = (
        (
            frame_now
            -
            frame_prev
        )
        /
        total_frame_distance
    )


    prev_x = float(
        df.loc[
            previous_anchor,
            "pitch_x_m"
        ]
    )

    prev_y = float(
        df.loc[
            previous_anchor,
            "pitch_y_m"
        ]
    )


    next_x = float(
        df.loc[
            next_anchor,
            "pitch_x_m"
        ]
    )

    next_y = float(
        df.loc[
            next_anchor,
            "pitch_y_m"
        ]
    )


    expected_x = (
        prev_x
        +
        alpha
        *
        (
            next_x
            -
            prev_x
        )
    )


    expected_y = (
        prev_y
        +
        alpha
        *
        (
            next_y
            -
            prev_y
        )
    )


    current_x = float(
        df.loc[
            i,
            "pitch_x_m"
        ]
    )

    current_y = float(
        df.loc[
            i,
            "pitch_y_m"
        ]
    )


    residual = float(
        np.sqrt(
            (
                current_x
                -
                expected_x
            ) ** 2
            +
            (
                current_y
                -
                expected_y
            ) ** 2
        )
    )


    df.loc[
        i,
        "temporal_residual_m"
    ] = residual


    # ========================================================
    # Validate medium-confidence observation
    # ========================================================

    if (
        residual
        <= MAX_MEDIUM_RESIDUAL_M
    ):

        df.loc[
            i,
            "trusted_observed"
        ] = True

        df.loc[
            i,
            "validation_reason"
        ] = "medium_conf_temporally_supported"

        medium_supported_count += 1

    else:

        df.loc[
            i,
            "validation_reason"
        ] = "rejected_temporal_residual"

        medium_rejected_count += 1


# ============================================================
# Copy trusted direct observations
# ============================================================

trusted_direct = (
    df["trusted_observed"]
)


df.loc[
    trusted_direct,
    "trusted_ball"
] = True


df.loc[
    trusted_direct,
    "pitch_x_trusted_m"
] = df.loc[
    trusted_direct,
    "pitch_x_m"
]


df.loc[
    trusted_direct,
    "pitch_y_trusted_m"
] = df.loc[
    trusted_direct,
    "pitch_y_m"
]


df.loc[
    trusted_direct,
    "image_x_trusted"
] = df.loc[
    trusted_direct,
    "image_x"
]


df.loc[
    trusted_direct,
    "image_y_trusted"
] = df.loc[
    trusted_direct,
    "image_y"
]


# ============================================================
# PASS 2
# Conservative interpolation
#
# Only fills small gaps BETWEEN trusted direct observations.
# ============================================================

interpolated_count = 0

n = len(df)

i = 0


while i < n:

    if bool(
        df.loc[
            i,
            "trusted_observed"
        ]
    ):

        i += 1
        continue


    gap_start = i


    while (
        i < n
        and
        not bool(
            df.loc[
                i,
                "trusted_observed"
            ]
        )
    ):

        i += 1


    gap_end = (
        i - 1
    )


    gap_length = (
        gap_end
        -
        gap_start
        +
        1
    )


    previous_index = (
        gap_start - 1
    )


    next_index = i


    # ========================================================
    # Basic interpolation requirements
    # ========================================================

    if (
        gap_length
        > MAX_INTERPOLATION_GAP
        or
        previous_index < 0
        or
        next_index >= n
    ):

        continue


    if not (
        bool(
            df.loc[
                previous_index,
                "trusted_observed"
            ]
        )
        and
        bool(
            df.loc[
                next_index,
                "trusted_observed"
            ]
        )
    ):

        continue


    previous_segment = df.loc[
        previous_index,
        "segment_id"
    ]


    next_segment = df.loc[
        next_index,
        "segment_id"
    ]


    # Do not bridge trajectory segments
    if (
        pd.isna(previous_segment)
        or
        pd.isna(next_segment)
        or
        previous_segment
        !=
        next_segment
    ):

        continue


    # ========================================================
    # Endpoint motion sanity
    # ========================================================

    frame_prev = float(
        df.loc[
            previous_index,
            "frame"
        ]
    )


    frame_next = float(
        df.loc[
            next_index,
            "frame"
        ]
    )


    dt = (
        frame_next
        -
        frame_prev
    ) / FPS


    if dt <= 0:

        continue


    x_prev = float(
        df.loc[
            previous_index,
            "pitch_x_m"
        ]
    )


    y_prev = float(
        df.loc[
            previous_index,
            "pitch_y_m"
        ]
    )


    x_next = float(
        df.loc[
            next_index,
            "pitch_x_m"
        ]
    )


    y_next = float(
        df.loc[
            next_index,
            "pitch_y_m"
        ]
    )


    endpoint_distance = float(
        np.sqrt(
            (
                x_next
                -
                x_prev
            ) ** 2
            +
            (
                y_next
                -
                y_prev
            ) ** 2
        )
    )


    endpoint_speed = (
        endpoint_distance
        /
        dt
    )


    # Too fast -> do not guess intermediate positions
    if (
        endpoint_speed
        >
        MAX_INTERPOLATION_SPEED
    ):

        continue


    # ========================================================
    # Image-space endpoints
    # ========================================================

    ix_prev = float(
        df.loc[
            previous_index,
            "image_x"
        ]
    )


    iy_prev = float(
        df.loc[
            previous_index,
            "image_y"
        ]
    )


    ix_next = float(
        df.loc[
            next_index,
            "image_x"
        ]
    )


    iy_next = float(
        df.loc[
            next_index,
            "image_y"
        ]
    )


    # ========================================================
    # Fill gap
    # ========================================================

    total_steps = (
        gap_length
        +
        1
    )


    for offset in range(
        1,
        gap_length + 1
    ):

        row_index = (
            previous_index
            +
            offset
        )


        alpha = (
            offset
            /
            total_steps
        )


        df.loc[
            row_index,
            "pitch_x_trusted_m"
        ] = (
            x_prev
            +
            alpha
            *
            (
                x_next
                -
                x_prev
            )
        )


        df.loc[
            row_index,
            "pitch_y_trusted_m"
        ] = (
            y_prev
            +
            alpha
            *
            (
                y_next
                -
                y_prev
            )
        )


        df.loc[
            row_index,
            "image_x_trusted"
        ] = (
            ix_prev
            +
            alpha
            *
            (
                ix_next
                -
                ix_prev
            )
        )


        df.loc[
            row_index,
            "image_y_trusted"
        ] = (
            iy_prev
            +
            alpha
            *
            (
                iy_next
                -
                iy_prev
            )
        )


        df.loc[
            row_index,
            "trusted_ball"
        ] = True


        df.loc[
            row_index,
            "interpolated"
        ] = True


        df.loc[
            row_index,
            "validation_reason"
        ] = "interpolated_short_gap"


        interpolated_count += 1


# ============================================================
# PASS 3
# Recalculate trusted trajectory speed
# ============================================================

df["trusted_frame_gap"] = np.nan

df["trusted_dt"] = np.nan

df["trusted_speed_mps"] = np.nan


trusted_indices = df.index[
    df["trusted_ball"]
].tolist()


previous_index = None


for current_index in trusted_indices:

    if previous_index is None:

        previous_index = current_index

        continue


    frame_prev = float(
        df.loc[
            previous_index,
            "frame"
        ]
    )


    frame_now = float(
        df.loc[
            current_index,
            "frame"
        ]
    )


    frame_gap = (
        frame_now
        -
        frame_prev
    )


    dt = (
        frame_gap
        /
        FPS
    )


    if dt <= 0:

        previous_index = current_index

        continue


    x_prev = float(
        df.loc[
            previous_index,
            "pitch_x_trusted_m"
        ]
    )


    y_prev = float(
        df.loc[
            previous_index,
            "pitch_y_trusted_m"
        ]
    )


    x_now = float(
        df.loc[
            current_index,
            "pitch_x_trusted_m"
        ]
    )


    y_now = float(
        df.loc[
            current_index,
            "pitch_y_trusted_m"
        ]
    )


    distance = float(
        np.sqrt(
            (
                x_now
                -
                x_prev
            ) ** 2
            +
            (
                y_now
                -
                y_prev
            ) ** 2
        )
    )


    speed = (
        distance
        /
        dt
    )


    df.loc[
        current_index,
        "trusted_frame_gap"
    ] = frame_gap


    df.loc[
        current_index,
        "trusted_dt"
    ] = dt


    df.loc[
        current_index,
        "trusted_speed_mps"
    ] = speed


    previous_index = current_index


# ============================================================
# Quality label
# ============================================================

def quality_label(row):

    if row["interpolated"]:

        return "interpolated"


    if row["trusted_observed"]:

        speed = row[
            "trusted_speed_mps"
        ]


        if (
            pd.notna(speed)
            and
            speed > HIGH_SPEED_FLAG
        ):

            return "trusted_high_speed"


        if (
            row["confidence"]
            >= HIGH_CONF
        ):

            return "trusted_high"


        return "trusted_supported"


    if row["observed"]:

        return "rejected"


    return "missing"


df["ball_quality"] = df.apply(
    quality_label,
    axis=1
)


# ============================================================
# Save CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Remaining gaps
# ============================================================

remaining_gaps = []

current_gap = 0


for trusted in df[
    "trusted_ball"
]:

    if not trusted:

        current_gap += 1

    else:

        if current_gap > 0:

            remaining_gaps.append(
                current_gap
            )

            current_gap = 0


if current_gap > 0:

    remaining_gaps.append(
        current_gap
    )


longest_remaining_gap = (
    max(remaining_gaps)
    if remaining_gaps
    else 0
)


# ============================================================
# Trusted speed statistics
# ============================================================

trusted_speeds = (
    df[
        "trusted_speed_mps"
    ]
    .replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )
    .dropna()
)


# ============================================================
# QA VIDEO
#
# Green  = trusted observed
# Cyan   = interpolated
# Red    = V3 observation rejected by V3.1
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        f"Cannot open video: {VIDEO_PATH}"
    )


fps_video = cap.get(
    cv2.CAP_PROP_FPS
)


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps_video,
    (
        width,
        height
    )
)


frame_idx = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    if frame_idx >= len(df):

        break


    row = df.iloc[
        frame_idx
    ]


    quality = row[
        "ball_quality"
    ]


    # ========================================================
    # Rejected V3 observation
    # ========================================================

    if (
        quality == "rejected"
        and
        pd.notna(
            row["image_x"]
        )
        and
        pd.notna(
            row["image_y"]
        )
    ):

        x = int(
            row["image_x"]
        )

        y = int(
            row["image_y"]
        )


        cv2.circle(
            frame,
            (
                x,
                y
            ),
            12,
            (
                0,
                0,
                255
            ),
            3
        )


        conf = (
            row["confidence"]
            if pd.notna(
                row["confidence"]
            )
            else 0.0
        )


        cv2.putText(
            frame,
            (
                f"REJECT "
                f"conf={conf:.2f}"
            ),
            (
                x + 14,
                y - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Trusted / interpolated
    # ========================================================

    if bool(
        row["trusted_ball"]
    ):

        x = int(
            row["image_x_trusted"]
        )

        y = int(
            row["image_y_trusted"]
        )


        if (
            quality
            ==
            "interpolated"
        ):

            color = (
                255,
                255,
                0
            )

            text = (
                "BALL INTERP"
            )

        elif (
            quality
            ==
            "trusted_high_speed"
        ):

            color = (
                0,
                165,
                255
            )

            text = (
                "BALL TRUSTED / FAST"
            )

        elif (
            quality
            ==
            "trusted_supported"
        ):

            color = (
                0,
                255,
                180
            )

            text = (
                "BALL SUPPORTED"
            )

        else:

            color = (
                0,
                255,
                0
            )

            text = (
                "BALL TRUSTED"
            )


        cv2.circle(
            frame,
            (
                x,
                y
            ),
            9,
            color,
            -1
        )


        cv2.putText(
            frame,
            text,
            (
                x + 12,
                y - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # Status
    # ========================================================

    status = (
        f"V3.1 quality: {quality}"
    )


    cv2.putText(
        frame,
        status,
        (
            20,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (
            255,
            255,
            255
        ),
        2,
        cv2.LINE_AA
    )


    writer.write(
        frame
    )


    frame_idx += 1


cap.release()

writer.release()


# ============================================================
# Summary
# ============================================================

raw_accepted = int(
    df["observed"].sum()
)


trusted_direct_count = int(
    df["trusted_observed"].sum()
)


trusted_total_count = int(
    df["trusted_ball"].sum()
)


raw_coverage = (
    raw_accepted
    /
    len(df)
    *
    100.0
)


trusted_coverage = (
    trusted_total_count
    /
    len(df)
    *
    100.0
)


print("")
print(
    "========================================"
)

print(
    "BALL VALIDATION V3.1"
)

print(
    "========================================"
)

print("")

print(
    "Frames:",
    len(df)
)

print("")

print(
    "V3 accepted observations:",
    raw_accepted
)

print(
    "V3 raw coverage:",
    f"{raw_coverage:.1f}%"
)

print("")

print(
    "High-confidence trusted:",
    high_conf_count
)

print(
    "Medium-confidence supported:",
    medium_supported_count
)

print(
    "Medium-confidence rejected:",
    medium_rejected_count
)

print(
    "Very-low-confidence rejected:",
    very_low_rejected_count
)

print("")

print(
    "Trusted direct observations:",
    trusted_direct_count
)

print(
    "Interpolated frames:",
    interpolated_count
)

print(
    "Final trusted frames:",
    trusted_total_count
)

print(
    "Final trusted coverage:",
    f"{trusted_coverage:.1f}%"
)

print("")

print(
    "Remaining missing frames:",
    int(
        (
            ~df["trusted_ball"]
        ).sum()
    )
)

print(
    "Longest remaining gap:",
    longest_remaining_gap,
    "frames"
)

print("")


if len(trusted_speeds) > 0:

    print(
        "Trusted median speed:",
        f"{trusted_speeds.median():.2f} m/s"
    )

    print(
        "Trusted P95 speed:",
        f"{trusted_speeds.quantile(0.95):.2f} m/s"
    )

    print(
        "Trusted P99 speed:",
        f"{trusted_speeds.quantile(0.99):.2f} m/s"
    )

    print(
        "Trusted max speed:",
        f"{trusted_speeds.max():.2f} m/s"
    )


print("")
print(
    "========================================"
)

print(
    "QUALITY COUNTS"
)

print(
    "========================================"
)

print("")

print(
    df["ball_quality"]
    .value_counts()
    .to_string()
)


# ============================================================
# Print rejected observations for manual QA
# ============================================================

rejected = df[
    df["ball_quality"]
    ==
    "rejected"
].copy()


print("")
print(
    "========================================"
)

print(
    "REJECTED OBSERVATIONS"
)

print(
    "========================================"
)


if len(rejected) == 0:

    print("")
    print(
        "None"
    )

else:

    print("")

    print(
        rejected[
            [
                "frame",
                "time_sec",
                "confidence",
                "ball_source",
                "validation_reason",
                "temporal_residual_m"
            ]
        ]
        .head(50)
        .round(3)
        .to_string(
            index=False
        )
    )


print("")
print(
    "========================================"
)

print(
    "DONE"
)

print(
    "========================================"
)

print("")

print(
    "CSV:",
    OUTPUT_CSV
)

print(
    "QA Video:",
    OUTPUT_VIDEO
)
