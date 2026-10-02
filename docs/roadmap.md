# Development Roadmap

[Back to README](../README.md).


The next stage is to turn the existing tracking, event, and tactical outputs into traceable coaching observations within this repository.

The milestones below describe planned work, not currently implemented capabilities. Version numbers are proposed release targets, subject to data availability and validation.

| Stage | Status | Scope |
| --- | --- | --- |
| v4.1.0 — CV and tactical analytics | Implemented; validated on one sample video; see README | Player/ball tracking, possession states, event extraction, identity QA, and team-level tactical metrics. |
| v5.0.0 — Coach Report V1 | Planned next | Deterministic feature extraction, rule-based observations, and JSON/Markdown reports with traceable evidence. |
| v5.1.0 — API and LLM-assisted reporting | Planned after report validation | FastAPI endpoints, Pydantic schemas, and optional LLM wording of verified observations. |
| v6.0.0 — Multi-match analysis and retrieval | Future exploration | Comparable multi-match records, structured statistical queries, and evidence retrieval for reporting or scouting questions. |

### Coach Report V1

Start with a report specification mapping each question to available input columns, metric definitions, evidence, and validation rules. Initial reports will describe the analyzed clip and valid observations, without implying full-match coverage.

Planned report questions:

- How is observed possession distributed between teams, and where do possession phases begin and end?
- How do team width, length, and compactness differ across in-possession and out-of-possession phases?
- How do available opponent-spacing measurements change around detected passes and receptions?
- Which supported event sequences coincide with the largest measured team-shape changes?
- Which observations have sufficient data coverage to include in a coaching report?

Planned outputs are structured JSON and deterministic Markdown reports covering clip overview, possession, team shape, spatial relationships, key sequences, and supported observations.

Release criteria:

- Every observation references its source run, strict or identity-aware branch, frame/time range, metric values, and relevant event IDs when applicable.
- Metric units, temporal windows, valid-frame counts, and coverage denominators are explicit.
- Missing or insufficient evidence is reported rather than filled with a tactical claim; coverage gaps are not treated as continuous observed play.
- Summaries reproduce the underlying measurements and preserve existing regression boundaries.
- Observed associations are distinguished from causal explanations or coaching recommendations. Event labels remain pipeline detections unless independently reviewed.
- Representative reports and checks for missing data, phase boundaries, and evidence references accompany the release.

### API and LLM-Assisted Reporting

Once the deterministic report is validated, expose it through a typed API and evaluate an optional LLM presentation layer.

Numerical aggregation and evidence selection will remain in code. Generated text must preserve the supplied measurements, uncertainty, and evidence references, with a deterministic fallback when generation or validation fails.

FastAPI, Pydantic, and LLM integration are planned additions, not current project dependencies or features.

### Multi-Match Analysis and Retrieval

Before cross-match comparison, establish consistent run metadata, metric definitions, coordinate conventions, and coverage rules across multiple validated videos.

Player-level comparisons additionally require verified identity mapping beyond the current team-assignment corrections.

Use structured queries for numerical comparisons. Evaluate RAG only when a useful collection of reports or annotated sequences exists for evidence retrieval. Vector search, scouting assistance, and frameworks such as LangChain remain exploratory choices.

---
