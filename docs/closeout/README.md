# Football CV Analytics — Project Closeout & Interview Report

**Status: prototype scope closed on 2026-10-15; remaining limitations are documented below.**

Baseline output snapshot: [Coach report](coach_report_v1.md) · [Structured evidence](coach_report_v1.json).

**專案收尾／面試展示版 · 15 October 2026**

Repository: `football-cv-analytics`

Baseline: supplied source corresponding to commit `a8b2889`; run `pipeline_v4_fresh`; tactical branch `identity_aware`.

This document closes the small prototype at a defined scope. It does not certify production readiness. The numerical results below were reproduced from the supplied baseline code and run CSVs. Experimental identity/control corrections are excluded from these results.

本文件將小專案收尾為可展示的原型。下列數值使用已合併版本的程式與提供的 CSV 重現；尚未完成回歸驗證的修正，不混入主結果。

## 1. English report — clip-level tactical observations

### Purpose and method

The prototype converts football video observations into structured tracking, possession states, event candidates and a draft coach report. This case uses a 30-second clip at 25 frames per second. The analysis contains 747 frames, numbered 3–749. It describes this clip only.

The report combines computer-vision outputs with geometric calculations and rule-based temporal processing. A frame is eligible for whole-team tactical analysis only when the selected dataset contains ten tracked outfield players from each team. This eligibility rule checks availability; it does not prove that every identity or coordinate is correct.

### Data coverage

| Measure | Result | Interpretation |
|---|---:|---|
| Analyzed frames | 747 | Observations from one short clip |
| Frames assigned to Team A or B | 407 / 747, 54.5% | Team assignment was available under the pipeline rules |
| Team A / Team B frames | 274 / 133 | A represents 67.3% of the known A/B assignments |
| Unknown / contested frames | 338 / 2 | Retained explicitly rather than forced into a team |
| Complete tactical frames | 405 / 747, 54.2% | Frames eligible for the whole-team shape calculation |
| Complete frames during A / B possession | 219 / 0 | An in-possession versus out-of-possession comparison is unavailable |

The 67.3% figure is conditional on known A/B assignments. It is not full-match possession and is not 67.3% of all analyzed frames. Consecutive video frames are also correlated; 747 frames do not represent 747 independent tactical situations.

### Finding 1 — Team A occupied a larger outfield footprint in the selected A-possession frames

| Median metric during Team A possession | Team A | Team B |
|---|---:|---:|
| Width | 39.078 m | 32.020 m |
| Length | 35.849 m | 20.932 m |
| Compactness distance | 22.118 m | 14.642 m |

These results use the same 219 complete frames for both teams. Width and length are the 90th–10th percentile spans of the corresponding pitch coordinates, calculated per frame and then summarized by the median. Compactness is the mean player distance from the team centroid per frame, summarized by the median; a smaller value indicates a tighter tracked group under this definition.

**Tactical interpretation:** Team A's tracked outfield players were more spread out, while Team B's were more compact in this selected sample. This is consistent with a wider attacking shape against a compact defending group. It does not establish the teams' intended formations, pressing strategy, or whether either shape was effective.

**Coach-facing follow-up:** Review the complete A-possession sequences to check whether the additional width produced usable passing options. The current aggregates cannot answer that question alone.

### Finding 2 — Two detected passing sequences coincided with increased inter-team spacing

| Detected sequence | Pre-pass median | Post-reception median | Difference |
|---|---:|---:|---:|
| Pass 27 → reception 28; reception frame 443 | 4.970 m | 6.676 m | +1.706 m |
| Pass 29 → reception 30; reception frame 540 | 7.723 m | 9.409 m | +1.686 m |

For the first sequence, the comparison uses frames 402–406 and 444–448. For the second, it uses frames 487–491 and 541–545. Each window contains five frames, equivalent to 0.20 seconds of frame observations at 25 fps.

The metric is the per-frame median of nearest-opponent distances across both teams, summarized across each window. It is an aggregate spacing measure, not the receiver's pressure. The changes are observed associations; they do not prove that the passes caused defensive separation or improved the attack. Other movement and the time elapsed during each pass can also affect the comparison.

**Coach-facing follow-up:** Inspect the two receptions in the video to assess the receiver's immediate options and nearby defenders. Do not label the sequences as successful pressure relief solely from these numbers.

### Finding 3 — The evidence does not support an attack-versus-defence comparison

The baseline contains no complete tactical frames during Team B's assigned possession. Therefore neither team's in-possession versus out-of-possession shape comparison is reported. The longest automatically detected Team A phase spans frames 401–611: 211 frame observations, or approximately 8.44 seconds. This is a local sequence, not evidence of sustained match dominance.

### Conclusion for this clip

The prototype supplies reproducible descriptive observations and directs a reviewer to specific video intervals. Its current value is to assist a coach or analyst in reviewing evidence. It does not yet support player recruitment decisions, full-match tactical conclusions, or claims of validated detection accuracy.

## 2. 中文解讀：你需要真正講懂的內容

**目前完成的是什麼？** 已經把影像分析輸出接到控球／事件判定、戰術指標與 Markdown／JSON 報告。報告能追溯到幀區間，資料不足時會停止產生某些比較。

**這段影片能講什麼足球意義？** 在選到的 A 隊控球片段，A 隊的場上球員分布較寬、較長，B 隊較集中。這可以形成「A 是否透過寬度創造傳球選項」的影片檢視問題；現在的資料不足以證明戰術目的或成效。

**為什麼不能直接說 A 控球率是 67.3%？** 分母是 407 個已知 A／B 歸屬幀。338 個 Unknown 不在分母中。A 佔全部分析幀約 36.7%，但 Unknown 中仍可能有 A 控球，所以也不能把 36.7% 當成完整真實控球率。

**為什麼不能說傳球讓接球者壓力下降？** 目前距離指標涵蓋兩隊全部選入的球員，並非只量接球者；前後差異也沒有排除其他因素。

**資料多一點就會全部好嗎？** 不會。較長、較多樣的資料能改善分析範圍與評估；漏偵測、追蹤斷裂、身分切換、角色混淆及篩選規則，仍需分別處理。沒有拍到的位置，也不能只靠增加同類影片就直接還原。

## 3. Known limitations and closeout decisions

| Issue | Evidence / effect | Decision for this release |
|---|---|---|
| Short-track filtering | A 20-second track-span rule excludes a short A-team fragment; baseline B-possession tactical coverage is zero | Disclose selection bias; do not publish the unsupported comparison |
| Role and identity continuity | Video/data review identified a goalkeeper-class inconsistency and a likely same-player track fragment | Record the case; do not claim automatic identity repair is complete |
| Conservative possession recovery | Unknown includes uncontrolled transit, missing ball observations and cases where later control is not accepted early enough | Keep Unknown; do not present every unknown frame as an error |
| Camera visibility and tracking gaps | A missing player may be off-camera, occluded, misclassified or missed by the detector | Record absence cause as undetermined when evidence cannot distinguish it |
| Limited validation | Existing eight report/event-window tests pass; no independent, fully labelled detection benchmark has been supplied | Report software test results separately from model accuracy |
| Clip scope | One 30-second clip; selected frames are temporally correlated | No season-level or recruitment conclusions |

The separate experimental work copy produced 558 complete tactical frames instead of 405, using reviewed identity/role annotations, and changed seven Unknown frames to Team A under a temporal-control heuristic. These are preliminary case results, not improvements established on an independent test set. They are not included in this release and have not been merged into the user's repository.

**收尾範圍：**保留目前可重現的原型、這份報告與已知限制。暫停新增追蹤修補、API、LLM／RAG 或完整平台功能。後續工作以面試準備為主。

## 4. Interview explanation — concise English version

The project is a football video analytics prototype. It takes tracking and ball observations, applies temporal rules to estimate possession and event sequences, and generates a coach report with evidence frames.

In the current clip, the system can describe team shape and compare aggregate spacing around selected passing sequences. The main challenge is deciding which observations are reliable enough to support a football conclusion.

For example, the baseline had no complete tactical frames during Team B possession. I investigated the data selection process and found that short-track filtering was part of the problem. Camera visibility and tracking errors are separate sources of missing data, so simply relaxing the player-count requirement would not be a reliable fix.

The report preserves unknown states and marks unsupported comparisons as unavailable. The next step is to use broader event data for match analysis and a labelled evaluation set for the video pipeline. I would evaluate a change by its correctness and downstream usefulness, not just by the number of frames it labels.

**使用方式：**以上是已完成工作的說明底稿。介紹個人貢獻時，依你真正理解、執行與驗證的部分調整；使用 AI 協助寫程式不等於自行訓練了所有偵測模型。

## 5. Reproduction and repository handoff

Run from the existing `football-cv-analytics` repository using its activated environment:

```bash
python -m unittest discover -s reports/tests -v
python reports/coach_report_v1.py --run-dir runs/pipeline_v4_fresh --branch identity_aware --include-event-windows --output-dir runs/pipeline_v4_fresh/reports_closeout
```

The report command refuses to overwrite an existing report unless `--overwrite` is explicitly supplied. Run it once, or choose another output folder if `reports_closeout` already exists. This is report generation from existing CSVs; it does not rerun the video detector.

This closeout is stored at `docs/closeout/README.md`. Keep future experiments in this same repository. The two adjacent sample files preserve the baseline report; they are not regenerated when a later run changes.

**Source files used for the numerical report:**

- `possession/data/possession_v6_frames.csv` and `possession_v6_episodes.csv`
- `analytics/outputs/team_tactical_metrics_frame_identity_aware.csv`
- `analytics/outputs/tactical_coverage_comparison.csv`
- `analytics/outputs/inter_team_spatial_metrics_frame_identity_aware.csv`
- `data/football_events_v1.csv`
- Baseline `reports/coach_report_v1.py`, `reports/event_windows.py` and metric implementations supplied by the user

No StatsBomb data was used to produce the results in this report.
