# Football CV Analytics

An end-to-end computer vision pipeline for converting football match video into structured player tracking, ball tracking, possession states, football events, and team-level tactical analytics.

The project is designed as a modular pipeline rather than a single inference script. It separates player tracking, trajectory cleaning, ball tracking, possession reasoning, event extraction, identity quality control, and tactical analysis into independently testable stages.

---

## Results at a Glance

### End-to-End Football Tracking

<p align="center">
  <img src="docs/assets/radar_tracking_frame.jpg" width="900" alt="Football player tracking and pitch radar visualization">
</p>

<p align="center">
  <em>Video-based player tracking with pitch projection and radar visualization.</em>
</p>

### Tactical Analytics

| Identity-Aware Team Shape | Average Positions |
| --- | --- |
| <img src="docs/assets/team_shape_snapshot_identity_aware.png" width="430" alt="Identity-aware team shape snapshot"> | <img src="docs/assets/average_positions.png" width="430" alt="Average player positions"> |
| Representative 10v10 team structure after high-confidence team-assignment correction. | Spatial summary of player positioning on the pitch. |

<p align="center">
  <img src="docs/assets/team_compactness_over_time_identity_aware.png" width="900" alt="Team compactness over time">
</p>

<p align="center">
  <em>Identity-aware team compactness over time for temporal tactical analysis.</em>
</p>

The full pipeline connects computer vision and football analytics:

**video → player tracking → trajectory cleaning → ball tracking → possession reasoning → event extraction → tactical analysis**

> Identity-aware analytics only corrects high-confidence team-assignment inconsistencies for team-level analysis. It does not claim recovery of physical player identity.

---

## Overview

The pipeline converts a football match video into structured tracking, event, and tactical data.

Main outputs include:

- Player tracking and pitch projection
- Cleaned player trajectories
- Ball tracking
- Possession and ball-motion states
- Football events
- Team-level tactical metrics and visualizations

The two public entry points are:

    python run_pipeline.py
    python run_analytics.py

`run_pipeline.py` runs the canonical computer-vision pipeline. `run_analytics.py` performs optional downstream tactical analysis.

## Pipeline

```text
Football Video
      │
      ▼
Player Tracking
      │
      ▼
Trajectory Cleaning
      │
      ▼
Ball Tracking
      │
      ▼
Possession / Ball State
      │
      ▼
Football Event Engine
      │
      ▼
Canonical Outputs
      │
      ▼
Optional Tactical Analytics
      │
      ├── Identity QA
      │
      ├── Strict Tactical Dataset
      │
      └── Identity-Aware Tactical Dataset
              │
              ├── Team Tactical Metrics
              ├── Inter-Team Spatial Metrics
              ├── Team Shape Over Time
              └── Representative Team Shape Snapshot
```

---

## Core Components

### Player Tracking

Canonical implementation:

`src/player/tracking.py`

The player pipeline performs detection, team assignment, tracking, and pitch projection.

The trajectory-cleaning stage is implemented in:

`src/player/trajectory.py`

Validated sample:

- 15,377 raw player observations
- 14,845 cleaned observations
- 22 tracked IDs

### Ball Tracking

Canonical implementation:

`src/ball/tracking.py`

The system combines multiple ball-detection signals and produces both raw and trusted ball trajectories.

Validated sample:

- 750 video frames
- 608 accepted raw detections
- 635 trusted usable frames
- 84.7% trusted ball coverage

### Possession and Ball State

Possession reasoning is implemented under:

`src/possession/`

The pipeline separates four concepts that should not be treated as equivalent:

- Controller
- Possession
- Last touch
- Ball motion

For example, a ball being in transit does not automatically imply a possession change.

### Football Event Engine

Canonical implementation:

`src/events/engine.py`

QA implementation:

`src/events/qa.py`

The validated sample produced 43 events:

- 28 PossessionEpisode
- 6 Dribble
- 4 Reception
- 3 Pass
- 1 AerialDuel
- 1 AirTouch

### Tactical Analytics

Tactical analysis is implemented under:

`src/analytics/`

It includes:

- Identity QA
- Strict complete 10v10 frame selection
- Identity-aware team-assignment correction
- Team centroid, length, width, compactness, and shape area
- Inter-team spatial metrics
- Team-shape visualization

Strict tactical coverage:

    323 / 747 frames (43.2%)

Identity-aware tactical coverage:

    405 / 747 frames (54.2%)

Additional complete tactical frames recovered:

    82

Identity-aware processing only applies high-confidence team-assignment corrections for team-level tactical analysis. It does not claim recovery of physical player identity.

For frames shared by the strict and identity-aware datasets, the tactical and inter-team metric implementations produced exact matching values.

## Installation

The validated development environment uses Python 3.9.6.

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Upgrade pip:

    python -m pip install --upgrade pip

Install the validated Python dependencies:

    python -m pip install -r requirements.txt

Verify the environment:

    python -m pip check

The validated dependency set includes NumPy, Pandas, SciPy, Matplotlib, OpenCV, scikit-learn, PyTorch, Ultralytics, Supervision, Sports, and PyYAML.

> Model weights are not included in the repository. Place the required model files under `models/` before running the main pipeline.

---

## Quick Verification

After installing the dependencies, the repository can be checked without model weights or input video:

    python scripts/smoke_test.py

The smoke test verifies:

- Required repository files
- Direct Python dependencies
- Python source compilation
- Main pipeline CLI
- Analytics CLI
- Representative sample-output schemas

A successful check ends with:

    SMOKE TEST: PASS

This is a lightweight repository check and does not run model inference or the full video pipeline.

---

## Quick Start

### 1. Install

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install -r requirements.txt

### 2. Prepare Models

The default configuration expects model weights under:

    models/

Model paths are defined in:

    configs/default.yaml

Model weights are intentionally not included in this repository.

### 3. Prepare a Video

For example:

    videos/match.mp4

Input videos are also excluded from Git.

### 4. Run the Main Pipeline

    python run_pipeline.py       --video videos/match.mp4       --run-name my_match

Useful options include:

    --config
    --device
    --resume
    --overwrite
    --skip-qa
    --dry-run

Example on Apple Silicon:

    python run_pipeline.py       --video videos/match.mp4       --run-name my_match       --device mps

### 5. Run Tactical Analytics

    python run_analytics.py       --run-dir runs/my_match       --overwrite

A custom cleaned tracking file can be supplied with:

    --tracking path/to/tracking_data_clean_v2.csv

## Sample Outputs

Small representative outputs are included under:

`examples/sample_outputs/`

These files allow the data structures produced by the pipeline to be inspected without running the full video pipeline.

| Output | Description |
| --- | --- |
| [tracking_sample.csv](examples/sample_outputs/tracking_sample.csv) | Small sample of cleaned player tracking with image coordinates, pitch coordinates, trajectory cleaning, and speed fields. |
| [football_events.csv](examples/sample_outputs/football_events.csv) | Complete event-engine output containing possession episodes, dribbles, receptions, passes, and other detected football events. |
| [tactical_coverage_comparison.csv](examples/sample_outputs/tactical_coverage_comparison.csv) | Comparison between strict and identity-aware complete 10v10 tactical coverage. |
| [team_tactical_metrics_summary_identity_aware.csv](examples/sample_outputs/team_tactical_metrics_summary_identity_aware.csv) | Team-level centroid, length, width, compactness, and shape-area statistics. |

For the validated sample:

- Strict tactical coverage: **323 / 747 frames (43.2%)**
- Identity-aware tactical coverage: **405 / 747 frames (54.2%)**
- Additional complete tactical frames recovered: **82**

The identity-aware outputs apply only high-confidence team-assignment corrections for team-level tactical analysis; they do not represent recovered physical player identities.

---

## Output Structure

A canonical run is stored under:

    runs/<run-name>/

Typical outputs include:

    runs/<run-name>/
    ├── data/
    ├── json/
    ├── videos/
    ├── logs/
    ├── possession/
    └── analytics/
        └── outputs/

Important outputs include cleaned tracking data, trusted ball tracking, football events, possession states, tactical metrics, and tactical visualizations.

Small representative CSV files are included in `examples/sample_outputs/` so the output schemas can be inspected without running model inference.

## Validation

The canonical pipeline was refactored only after regression boundaries were established.

Validated sample:

| Stage | Result |
| --- | --- |
| Raw player tracking | 15,377 rows |
| Cleaned player tracking | 14,845 rows / 22 tracks |
| Ball tracking | 750 frames |
| Trusted ball coverage | 635 / 750 frames |
| Possession output | 747 frames |
| Football events | 43 events |
| Strict tactical coverage | 323 / 747 frames |
| Identity-aware coverage | 405 / 747 frames |

The main tracking, ball, possession, and event stages passed exact regression checks during package restructuring.

For the 323 tactical frames shared by the strict and identity-aware branches:

    Team tactical metrics: max_diff = 0.0
    Inter-team spatial metrics: max_diff = 0.0

This verifies that identity-aware processing extends usable team-level tactical coverage without altering the original strict measurements on common frames.

## Repository Structure

    .
    ├── configs/
    ├── docs/
    │   └── assets/
    ├── examples/
    │   └── sample_outputs/
    ├── legacy/
    ├── scripts/
    │   └── smoke_test.py
    ├── src/
    │   ├── analytics/
    │   ├── ball/
    │   ├── events/
    │   ├── player/
    │   └── possession/
    ├── run_pipeline.py
    ├── run_analytics.py
    ├── requirements.txt
    └── README.md

Recommended entry points:

    run_pipeline.py
    run_analytics.py

Earlier experimental implementations are retained under `legacy/` and are not part of the recommended execution path.

## Configuration

Default configuration:

`configs/default.yaml`

It contains model paths, runtime device settings, trajectory-cleaning parameters, motion thresholds, and pipeline-version settings.

The runtime device defaults to `auto` and can be overridden with:

    --device

## Limitations

The current system has several important limitations.

- Tracking IDs should not be interpreted as verified physical player identities.
- Identity-aware analytics corrects high-confidence team-assignment inconsistencies only.
- Tactical analytics currently focuses on team-level geometry and spatial structure.
- Quantitative values shown in this README come from one validated sample video.
- These values are regression checks, not standardized benchmark results.
- Model weights are not included in the repository.

---

## License

This project is released under the [MIT License](LICENSE).

Model weights, input videos, datasets, and third-party dependencies are not distributed as part of this repository and remain subject to their respective licenses and terms.

---
