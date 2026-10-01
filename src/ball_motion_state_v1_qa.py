import argparse
import os

import cv2
import numpy as np
import pandas as pd


def txt(v, default="-"):
    if pd.isna(v):
        return default

    s = str(v).strip()

    if s.lower() in {
        "",
        "nan",
        "none",
    }:
        return default

    return s


def track_text(
    team,
    track_id,
):
    team = txt(
        team,
        None,
    )

    if (
        team is None
        or
        pd.isna(track_id)
    ):
        return "NONE"

    try:
        track_id = int(
            float(track_id)
        )

    except Exception:
        return "NONE"

    return (
        f"{team}#{track_id}"
    )


def find_xy_columns(
    df,
):
    candidates = [
        (
            "ball_x_px",
            "ball_y_px",
        ),
        (
            "ball_px_x",
            "ball_px_y",
        ),
        (
            "ball_image_x",
            "ball_image_y",
        ),
        (
            "ball_x",
            "ball_y",
        ),
        (
            "x_ball",
            "y_ball",
        ),
    ]

    for x_col, y_col in candidates:

        if (
            x_col in df.columns
            and
            y_col in df.columns
        ):

            return (
                x_col,
                y_col,
            )

    return (
        None,
        None,
    )


def draw_text(
    frame,
    text,
    x,
    y,
    color=(255, 255, 255),
    scale=0.62,
    thickness=1,
):
    cv2.putText(
        frame,
        text,
        (
            int(x),
            int(y),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (
            0,
            0,
            0,
        ),
        thickness + 3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (
            int(x),
            int(y),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def main():

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(
        "--video",
        default=(
            "videos/match.mp4"
        ),
    )


    parser.add_argument(
        "--frames",
        default=(
            "outputs/"
            "ball_motion_state_v1_frames.csv"
        ),
    )


    parser.add_argument(
        "--events",
        default=(
            "outputs/"
            "air_touch_v4_events.csv"
        ),
    )


    parser.add_argument(
        "--output",
        default=(
            "outputs/"
            "ball_motion_state_v1_qa.mp4"
        ),
    )


    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )


    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
    )


    args = parser.parse_args()


    # ========================================================
    # LOAD CSV
    # ========================================================

    motion = pd.read_csv(
        args.frames
    )


    events = pd.read_csv(
        args.events
    )


    motion[
        "frame"
    ] = pd.to_numeric(
        motion[
            "frame"
        ],
        errors="coerce",
    )


    motion = (
        motion
        .dropna(
            subset=[
                "frame"
            ]
        )
        .copy()
    )


    motion[
        "frame"
    ] = (
        motion[
            "frame"
        ]
        .astype(int)
    )


    motion_by_frame = (
        motion
        .set_index(
            "frame"
        )
    )


    # ========================================================
    # BALL COORDINATE
    # ========================================================

    ball_x_col, ball_y_col = (
        find_xy_columns(
            motion
        )
    )


    if (
        ball_x_col
        is not None
    ):

        print(
            "Ball image columns:",
            ball_x_col,
            ball_y_col,
        )

    else:

        print(
            "No ball image coordinate "
            "columns found."
        )


    # ========================================================
    # EVENT LOOKUP
    # ========================================================

    event_lookup = {}


    if len(
        events
    ) > 0:

        for _, event in (
            events.iterrows()
        ):

            try:

                start = int(
                    event[
                        "start_frame"
                    ]
                )

                end = int(
                    event[
                        "end_frame"
                    ]
                )

            except Exception:

                continue


            event_class = txt(
                event.get(
                    "final_class"
                )
            )


            participants = txt(
                event.get(
                    "participants"
                )
            )


            reason = txt(
                event.get(
                    "decision_reason"
                )
            )


            label = (
                f"{event_class} "
                f"[{participants}]"
            )


            for frame_id in range(
                start,
                end + 1,
            ):

                if (
                    frame_id
                    not in
                    event_lookup
                ):

                    event_lookup[
                        frame_id
                    ] = []


                event_lookup[
                    frame_id
                ].append(
                    {
                        "label":
                            label,

                        "reason":
                            reason,
                    }
                )


    # ========================================================
    # VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        args.video
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: "
            f"{args.video}"
        )


    fps = cap.get(
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


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    start_frame = max(
        0,
        args.start_frame,
    )


    if (
        args.end_frame
        <
        0
    ):

        end_frame = (
            total_frames - 1
        )

    else:

        end_frame = min(
            args.end_frame,
            total_frames - 1,
        )


    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )


    os.makedirs(
        os.path.dirname(
            args.output
        ),
        exist_ok=True,
    )


    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )


    writer = cv2.VideoWriter(
        args.output,
        fourcc,
        fps,
        (
            width,
            height,
        ),
    )


    if not writer.isOpened():

        raise RuntimeError(
            f"Cannot open writer: "
            f"{args.output}"
        )


    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        f"Rendering "
        f"{start_frame}"
        f" -> "
        f"{end_frame}"
    )


    # ========================================================
    # FRAME LOOP
    # ========================================================

    frame_id = start_frame


    while (
        frame_id
        <=
        end_frame
    ):

        ok, frame = (
            cap.read()
        )


        if not ok:

            break


        if (
            frame_id
            not in
            motion_by_frame.index
        ):

            writer.write(
                frame
            )

            frame_id += 1

            continue


        row = (
            motion_by_frame.loc[
                frame_id
            ]
        )


        if isinstance(
            row,
            pd.DataFrame,
        ):

            row = (
                row.iloc[0]
            )


        # ====================================================
        # CURRENT VALUES
        # ====================================================

        motion_state = txt(
            row.get(
                "motion_state_v1"
            )
        )


        flight = txt(
            row.get(
                "flight_evidence_v1"
            )
        )


        possession = txt(
            row.get(
                "possession_motion_v1"
            )
        )


        controller = track_text(
            row.get(
                "controller_team_motion_v1"
            ),
            row.get(
                "controller_id_motion_v1"
            ),
        )


        last_touch = track_text(
            row.get(
                "last_touch_team_motion_v1"
            ),
            row.get(
                "last_touch_id_motion_v1"
            ),
        )


        episode = txt(
            row.get(
                "motion_episode_id_v1"
            )
        )


        suppressed = bool(
            row.get(
                "controller_suppressed_by_motion_v1",
                False,
            )
        )


        # ====================================================
        # BASELINE COMPARISON
        # ====================================================

        baseline_state = txt(
            row.get(
                "ball_state_v5_3"
            )
        )


        baseline_possession = txt(
            row.get(
                "possession_v5_3"
            )
        )


        baseline_controller = (
            track_text(
                row.get(
                    "controller_team_v5_3"
                ),
                row.get(
                    "controller_id_v5_3"
                ),
            )
        )


        # ====================================================
        # PANEL
        # ====================================================

        panel_w = min(
            640,
            width - 20,
        )


        panel_h = 260


        overlay = (
            frame.copy()
        )


        cv2.rectangle(
            overlay,
            (
                10,
                10,
            ),
            (
                panel_w,
                panel_h,
            ),
            (
                0,
                0,
                0,
            ),
            -1,
        )


        frame = cv2.addWeighted(
            overlay,
            0.60,
            frame,
            0.40,
            0,
        )


        time_sec = (
            frame_id
            /
            fps
        )


        y = 38


        draw_text(
            frame,
            (
                f"Frame {frame_id} "
                f"| {time_sec:.2f}s "
                f"| Episode {episode}"
            ),
            25,
            y,
        )


        y += 32


        # Motion state highlight
        if (
            motion_state
            ==
            "AerialContest"
        ):

            state_color = (
                0,
                165,
                255,
            )


        elif (
            motion_state
            ==
            "AirTouchInTransit"
        ):

            state_color = (
                255,
                100,
                255,
            )


        elif (
            motion_state
            ==
            "DribbleGap"
        ):

            state_color = (
                255,
                255,
                0,
            )


        elif (
            "ConfirmedPass"
            in
            motion_state
        ):

            state_color = (
                100,
                255,
                100,
            )


        elif (
            motion_state
            in
            {
                "TransitUncontrolled",
                "TransitMissing",
            }
        ):

            state_color = (
                0,
                220,
                255,
            )


        else:

            state_color = (
                255,
                255,
                255,
            )


        draw_text(
            frame,
            (
                "MOTION: "
                f"{motion_state}"
            ),
            25,
            y,
            state_color,
            scale=0.72,
            thickness=2,
        )


        y += 32


        draw_text(
            frame,
            (
                "FLIGHT EVIDENCE: "
                f"{flight}"
            ),
            25,
            y,
        )


        y += 30


        draw_text(
            frame,
            (
                "POSSESSION: "
                f"{possession}"
            ),
            25,
            y,
        )


        y += 30


        draw_text(
            frame,
            (
                "CONTROLLER: "
                f"{controller}"
            ),
            25,
            y,
        )


        y += 30


        draw_text(
            frame,
            (
                "LAST TOUCH: "
                f"{last_touch}"
            ),
            25,
            y,
        )


        y += 30


        draw_text(
            frame,
            (
                "BASELINE V5.3: "
                f"{baseline_possession} / "
                f"{baseline_state} / "
                f"{baseline_controller}"
            ),
            25,
            y,
            (
                190,
                190,
                190,
            ),
            scale=0.52,
        )


        if suppressed:

            draw_text(
                frame,
                "CONTROLLER SUPPRESSED",
                width - 320,
                40,
                (
                    0,
                    0,
                    255,
                ),
                scale=0.65,
                thickness=2,
            )


        # ====================================================
        # PHYSICAL EVENT BANNER
        # ====================================================

        frame_events = (
            event_lookup.get(
                frame_id,
                [],
            )
        )


        event_y = (
            height - 60
        )


        for event in (
            frame_events[:2]
        ):

            draw_text(
                frame,
                (
                    "PHYSICAL EVENT: "
                    f"{event['label']}"
                ),
                25,
                event_y,
                (
                    255,
                    255,
                    0,
                ),
                scale=0.62,
                thickness=2,
            )

            event_y -= 28


        # ====================================================
        # OPTIONAL BALL DOT
        # ====================================================

        if (
            ball_x_col
            is not None
        ):

            bx = row.get(
                ball_x_col
            )

            by = row.get(
                ball_y_col
            )


            if (
                pd.notna(bx)
                and
                pd.notna(by)
            ):

                bx = int(
                    round(
                        float(bx)
                    )
                )

                by = int(
                    round(
                        float(by)
                    )
                )


                if (
                    0 <= bx < width
                    and
                    0 <= by < height
                ):

                    cv2.circle(
                        frame,
                        (
                            bx,
                            by,
                        ),
                        11,
                        (
                            0,
                            0,
                            255,
                        ),
                        2,
                    )


        writer.write(
            frame
        )


        frame_id += 1


    cap.release()

    writer.release()


    print("")
    print(
        "QA video saved:"
    )

    print(
        args.output
    )


if __name__ == "__main__":

    main()
