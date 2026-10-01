import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

VIDEO_PATH = "videos/match.mp4"

INPUT_CSV = "outputs/ball_tracking_ensemble.csv"

OUTPUT_CSV = "outputs/ball_tracking_ensemble_trusted.csv"
OUTPUT_VIDEO = "outputs/ball_tracking_ensemble_trusted.mp4"

os.makedirs("outputs", exist_ok=True)


# ============================================================
# Parameters
# ============================================================

FPS = 25.0


# SINGLE candidate must be stronger than DUAL
SINGLE_MIN_CONF = 0.50

SINGLE_MAX_IMAGE_RESIDUAL_PX = 55.0
SINGLE_MAX_PITCH_RESIDUAL_M = 3.5
SINGLE_MAX_SPEED_MPS = 50.0


# Reacquired SINGLE already passed two-frame confirmation,
# but require higher confidence.
SINGLE_REACQUIRE_MIN_CONF = 0.70


# Short interpolation only
MAX_INTERPOLATION_GAP = 3

MAX_INTERPOLATION_SPEED_MPS = 35.0


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
# Normalize
# ============================================================

df["accepted_bool"] = (
    df["accepted"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

df["segment_id"] = pd.to_numeric(
    df["segment_id"],
    errors="coerce"
)

df["confidence"] = pd.to_numeric(
    df["confidence"],
    errors="coerce"
)

df["image_residual_px"] = pd.to_numeric(
    df["image_residual_px"],
    errors="coerce"
)

df["pitch_residual_m"] = pd.to_numeric(
    df["pitch_residual_m"],
    errors="coerce"
)

df["estimated_speed_mps"] = pd.to_numeric(
    df["estimated_speed_mps"],
    errors="coerce"
)


# ============================================================
# Output columns
# ============================================================

df["ball_trusted_direct"] = False
df["ball_interpolated"] = False
df["ball_usable"] = False

df["ball_quality"] = "missing"
df["trust_reason"] = "missing"

df["trusted_segment_id"] = np.nan

df["image_x_trusted"] = np.nan
df["image_y_trusted"] = np.nan

df["pitch_x_trusted_m"] = np.nan
df["pitch_y_trusted_m"] = np.nan


# ============================================================
# PASS 1
# Direct observation validation
# ============================================================

dual_trusted_count = 0
single_trusted_count = 0
single_rejected_count = 0


for i in range(len(df)):

    if not df.loc[i, "accepted_bool"]:

        continue


    candidate_type = str(
        df.loc[i, "candidate_type"]
    )

    mode = str(
        df.loc[i, "mode"]
    )

    confidence = df.loc[
        i,
        "confidence"
    ]


    # ========================================================
    # DUAL
    #
    # Two independent models agree.
    # ========================================================

    if candidate_type == "dual":

        df.loc[
            i,
            "ball_trusted_direct"
        ] = True

        df.loc[
            i,
            "ball_quality"
        ] = "trusted_dual"

        df.loc[
            i,
            "trust_reason"
        ] = "cross_model_agreement"

        dual_trusted_count += 1

        continue


    # ========================================================
    # SINGLE
    # ========================================================

    if candidate_type not in [
        "old_single",
        "forza_single"
    ]:

        df.loc[
            i,
            "ball_quality"
        ] = "rejected"

        df.loc[
            i,
            "trust_reason"
        ] = "unknown_candidate_type"

        continue


    # --------------------------------------------------------
    # SINGLE reacquisition
    #
    # Already passed V ensemble's two-frame confirmation.
    # --------------------------------------------------------

    if mode == "reacquired":

        if (
            pd.notna(confidence)
            and
            confidence
            >= SINGLE_REACQUIRE_MIN_CONF
        ):

            df.loc[
                i,
                "ball_trusted_direct"
            ] = True

            df.loc[
                i,
                "ball_quality"
            ] = "trusted_single"

            df.loc[
                i,
                "trust_reason"
            ] = "single_confirmed_reacquisition"

            single_trusted_count += 1

        else:

            df.loc[
                i,
                "ball_quality"
            ] = "rejected"

            df.loc[
                i,
                "trust_reason"
            ] = "single_reacquire_low_conf"

            single_rejected_count += 1

        continue


    # --------------------------------------------------------
    # Normal tracked SINGLE
    # --------------------------------------------------------

    image_residual = df.loc[
        i,
        "image_residual_px"
    ]

    pitch_residual = df.loc[
        i,
        "pitch_residual_m"
    ]

    speed = df.loc[
        i,
        "estimated_speed_mps"
    ]


    confidence_ok = (
        pd.notna(confidence)
        and
        confidence
        >= SINGLE_MIN_CONF
    )


    image_ok = (
        pd.notna(image_residual)
        and
        image_residual
        <= SINGLE_MAX_IMAGE_RESIDUAL_PX
    )


    pitch_ok = (
        pd.notna(pitch_residual)
        and
        pitch_residual
        <= SINGLE_MAX_PITCH_RESIDUAL_M
    )


    speed_ok = (
        pd.notna(speed)
        and
        speed
        <= SINGLE_MAX_SPEED_MPS
    )


    if (
        confidence_ok
        and
        image_ok
        and
        pitch_ok
        and
        speed_ok
    ):

        df.loc[
            i,
            "ball_trusted_direct"
        ] = True

        df.loc[
            i,
            "ball_quality"
        ] = "trusted_single"

        df.loc[
            i,
            "trust_reason"
        ] = "single_temporally_supported"

        single_trusted_count += 1

    else:

        df.loc[
            i,
            "ball_quality"
        ] = "rejected"

        df.loc[
            i,
            "trust_reason"
        ] = "single_failed_strict_validation"

        single_rejected_count += 1


# ============================================================
# Copy direct trusted observations
# ============================================================

mask = df[
    "ball_trusted_direct"
]


df.loc[
    mask,
    "ball_usable"
] = True


df.loc[
    mask,
    "trusted_segment_id"
] = df.loc[
    mask,
    "segment_id"
]


df.loc[
    mask,
    "image_x_trusted"
] = df.loc[
    mask,
    "image_x"
]


df.loc[
    mask,
    "image_y_trusted"
] = df.loc[
    mask,
    "image_y"
]


df.loc[
    mask,
    "pitch_x_trusted_m"
] = df.loc[
    mask,
    "pitch_x_m"
]


df.loc[
    mask,
    "pitch_y_trusted_m"
] = df.loc[
    mask,
    "pitch_y_m"
]


# ============================================================
# PASS 2
# Short-gap interpolation
# ============================================================

interpolated_count = 0

n = len(df)

i = 0


while i < n:

    if df.loc[
        i,
        "ball_trusted_direct"
    ]:

        i += 1
        continue


    gap_start = i


    while (
        i < n
        and
        not df.loc[
            i,
            "ball_trusted_direct"
        ]
    ):

        i += 1


    gap_end = i - 1

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


    if (
        gap_length
        >
        MAX_INTERPOLATION_GAP
    ):

        continue


    if (
        previous_index < 0
        or
        next_index >= n
    ):

        continue


    if not (
        df.loc[
            previous_index,
            "ball_trusted_direct"
        ]
        and
        df.loc[
            next_index,
            "ball_trusted_direct"
        ]
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


    # Never bridge different trajectory segments
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


    frame_prev = df.loc[
        previous_index,
        "frame"
    ]

    frame_next = df.loc[
        next_index,
        "frame"
    ]


    dt = (
        frame_next
        -
        frame_prev
    ) / FPS


    if dt <= 0:

        continue


    x_prev = df.loc[
        previous_index,
        "pitch_x_m"
    ]

    y_prev = df.loc[
        previous_index,
        "pitch_y_m"
    ]


    x_next = df.loc[
        next_index,
        "pitch_x_m"
    ]

    y_next = df.loc[
        next_index,
        "pitch_y_m"
    ]


    distance = float(
        np.hypot(
            x_next - x_prev,
            y_next - y_prev
        )
    )


    endpoint_speed = (
        distance / dt
    )


    if (
        endpoint_speed
        >
        MAX_INTERPOLATION_SPEED_MPS
    ):

        continue


    ix_prev = df.loc[
        previous_index,
        "image_x"
    ]

    iy_prev = df.loc[
        previous_index,
        "image_y"
    ]


    ix_next = df.loc[
        next_index,
        "image_x"
    ]

    iy_next = df.loc[
        next_index,
        "image_y"
    ]


    total_steps = (
        gap_length + 1
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
                x_next - x_prev
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
                y_next - y_prev
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
                ix_next - ix_prev
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
                iy_next - iy_prev
            )
        )


        df.loc[
            row_index,
            "trusted_segment_id"
        ] = previous_segment


        df.loc[
            row_index,
            "ball_usable"
        ] = True


        df.loc[
            row_index,
            "ball_interpolated"
        ] = True


        df.loc[
            row_index,
            "ball_quality"
        ] = "interpolated"


        df.loc[
            row_index,
            "trust_reason"
        ] = "short_gap_interpolation"


        interpolated_count += 1


# ============================================================
# PASS 3
# Trusted speed
# ============================================================

df["trusted_speed_mps"] = np.nan


trusted_df = df[
    df["ball_usable"]
].copy()


for segment_id, segment in trusted_df.groupby(
    "trusted_segment_id",
    dropna=True
):

    segment = (
        segment
        .sort_values("frame")
    )


    indices = segment.index.tolist()


    previous_index = None


    for current_index in indices:

        if previous_index is None:

            previous_index = current_index
            continue


        frame_gap = (
            df.loc[
                current_index,
                "frame"
            ]
            -
            df.loc[
                previous_index,
                "frame"
            ]
        )


        # Do not calculate across long gaps
        if (
            frame_gap <= 0
            or
            frame_gap > 5
        ):

            previous_index = current_index
            continue


        dt = (
            frame_gap / FPS
        )


        distance = float(
            np.hypot(

                df.loc[
                    current_index,
                    "pitch_x_trusted_m"
                ]
                -
                df.loc[
                    previous_index,
                    "pitch_x_trusted_m"
                ],

                df.loc[
                    current_index,
                    "pitch_y_trusted_m"
                ]
                -
                df.loc[
                    previous_index,
                    "pitch_y_trusted_m"
                ]
            )
        )


        df.loc[
            current_index,
            "trusted_speed_mps"
        ] = (
            distance / dt
        )


        previous_index = current_index


# ============================================================
# Remaining gaps
# ============================================================

gaps = []

gap = 0


for usable in df["ball_usable"]:

    if not usable:

        gap += 1

    else:

        if gap > 0:

            gaps.append(gap)
            gap = 0


if gap > 0:

    gaps.append(gap)


longest_gap = (
    max(gaps)
    if gaps
    else 0
)


# ============================================================
# Save CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# QA video
#
# Green  = DUAL
# Orange = trusted SINGLE
# Cyan   = interpolated
# Red    = ensemble accepted but rejected by trusted layer
# ============================================================

cap = cv2.VideoCapture(
    VIDEO_PATH
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

fps_video = cap.get(
    cv2.CAP_PROP_FPS
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


    # --------------------------------------------------------
    # Rejected accepted candidate
    # --------------------------------------------------------

    if (
        quality == "rejected"
        and
        row["accepted_bool"]
        and
        pd.notna(row["image_x"])
        and
        pd.notna(row["image_y"])
    ):

        x = int(
            row["image_x"]
        )

        y = int(
            row["image_y"]
        )


        cv2.circle(
            frame,
            (x, y),
            10,
            (
                0,
                0,
                255
            ),
            3
        )


        cv2.putText(
            frame,
            "REJECT",
            (
                x + 12,
                y - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (
                0,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Trusted / interpolation
    # --------------------------------------------------------

    if row["ball_usable"]:

        x = int(
            row[
                "image_x_trusted"
            ]
        )

        y = int(
            row[
                "image_y_trusted"
            ]
        )


        if quality == "trusted_dual":

            color = (
                0,
                255,
                0
            )

            label = "BALL DUAL"


        elif quality == "trusted_single":

            color = (
                0,
                165,
                255
            )

            label = "BALL SINGLE"


        else:

            color = (
                255,
                255,
                0
            )

            label = "BALL INTERP"


        cv2.circle(
            frame,
            (x, y),
            8,
            color,
            -1
        )


        cv2.putText(
            frame,
            label,
            (
                x + 10,
                y - 8
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA
        )


    cv2.putText(
        frame,
        (
            f"Trusted Ball | "
            f"{quality}"
        ),
        (
            20,
            35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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

usable_count = int(
    df["ball_usable"].sum()
)

usable_coverage = (
    usable_count
    /
    len(df)
    *
    100.0
)


speeds = (
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


print("")
print("========================================")
print("TRUSTED ENSEMBLE BALL SUMMARY")
print("========================================")
print("")

print(
    "Frames:",
    len(df)
)

print("")

print(
    "Trusted DUAL:",
    dual_trusted_count
)

print(
    "Trusted SINGLE:",
    single_trusted_count
)

print(
    "Rejected SINGLE:",
    single_rejected_count
)

print(
    "Interpolated:",
    interpolated_count
)

print("")

print(
    "Final usable frames:",
    usable_count
)

print(
    "Final usable coverage:",
    f"{usable_coverage:.1f}%"
)

print(
    "Missing / unknown:",
    int(
        (~df["ball_usable"]).sum()
    )
)

print(
    "Longest remaining gap:",
    longest_gap,
    "frames"
)


if len(speeds) > 0:

    print("")

    print(
        "Speed median:",
        f"{speeds.median():.2f} m/s"
    )

    print(
        "Speed P95:",
        f"{speeds.quantile(0.95):.2f} m/s"
    )

    print(
        "Speed P99:",
        f"{speeds.quantile(0.99):.2f} m/s"
    )

    print(
        "Speed max:",
        f"{speeds.max():.2f} m/s"
    )


print("")
print(
    df[
        "ball_quality"
    ]
    .value_counts()
    .to_string()
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
