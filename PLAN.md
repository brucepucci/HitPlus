# Hit+ ML Framework: Implementation Plan

## Context

Build a "Hit+" hitter-outcome model framework inspired by Stuff+/PLV but from the batter's perspective. The system is a collection of pitch-by-pitch submodels (swing decision, called strike, contact quality, etc.) that will eventually combine into a single Hit+ score. This plan focuses on building a robust, validated framework around the **first submodel only** — the Swing Decision Model — then making it trivial to add subsequent submodels.

The user's top priorities: **robust validation**, idempotent stateful pipeline, CLI-driven, Pythonic, simple models first. Uses **polars** (not polars) for all DataFrame operations.

**Data:** ~5.3M regular season pitches (2018-2025) in `data/mlb_stats.db` with excellent Statcast coverage (99.7%+ on velocity, location, movement, spin).

---

## First Submodel: Swing Decision Model

**Why start here:**
- Largest sample size — every pitch is labeled (swing vs take from `call_code`)
- Cleanest labels — direct observable, not a proxy
- Near-balanced classes (~50/50 swing/take)
- Strong domain validation available (batters swing more in zone, with 2 strikes, etc.)
- Foundation for Hit+ — swing decision is the first branching point in any PA outcome tree

**Target:** `is_swing = 1` if `call_code IN ('S','F','X','D','E','T','W','L','M')`, else `0`
- Exclude `*B` (pitch clock auto-ball) — batter had no agency
- Exclude `H` (hit by pitch) — batter had no swing/take decision

**Features (v1 — deliberately lean):**

| Feature | Source | Rationale |
|---------|--------|-----------|
| `plateX`, `plateZ` | pitch location | #1 and #2 swing predictors |
| `plateX_sq`, `plateZ_sq` | squared terms | Edge-of-zone nonlinearity for logistic |
| `sz_ratio` | `(plateZ - szBottom) / (szTop - szBottom)` | Normalized height (accounts for batter height) |
| `in_zone` | `1 if zone 1-9 else 0` | Binary zone indicator |
| `startSpeed` | release velocity | Faster = less swing |
| `pfxX`, `pfxZ` | horizontal/vertical movement | Break characteristics |
| `pitch_type_encoded` | `type_code` | One-hot or target-encoded |
| `balls`, `strikes` | count before pitch | Count context |
| `is_two_strikes` | `strikes == 2` | Explicit protect-the-plate feature |
| `is_rhb`, `is_rhp`, `same_hand` | handedness | Platoon context |

**Excluded from v1** (can measure lift later): outs, inning, score, runners, extension, plate time, spin rate/direction, batter identity.

**Temporal Split — Iterative Sizing Strategy:**

Start small for fast iteration, scale up once the framework and model are validated:

- **Dev mode (default):** Train on 2023 only (~718K pitches), test on 2024 (~700K). Fast feedback loops (~1.4M total).
- **Full mode (`--full`):** Train on 2018-2023 (~4.1M pitches), test on 2024 (~700K). Run once framework is proven.
- **Holdout:** 2025 (~710K) — never touched until final evaluation in either mode.
- **CV:** `TimeSeriesSplit` with folds on training data (2 folds in dev, 4 in full).

The `HitPlusConfig` has a `dev_mode: bool = True` flag. CLI: `hitplus run --model swing_decision --full` to override.

**Models:** Logistic Regression (interpretable baseline) + LightGBM (tree-based comparison). If logistic is within 1-2 AUC of LightGBM, prefer logistic.

---

## Project Structure

```
Hit+/
├── pyproject.toml              # uv project, black/ruff/pytest config
├── Makefile                    # make train, make test, make lint
├── data/
│   └── mlb_stats.db            # Existing, never modified
├── artifacts/                  # Git-ignored, all pipeline outputs
│   ├── datasets/swing_decision/
│   ├── models/swing_decision/
│   ├── validation/swing_decision/
│   └── plots/swing_decision/
├── src/hitplus/
│   ├── __init__.py
│   ├── cli.py                  # Click CLI entrypoint
│   ├── core/
│   │   ├── pipeline.py         # PipelineStep ABC, Pipeline orchestrator
│   │   ├── artifact.py         # ArtifactStore (manages artifacts/)
│   │   ├── config.py           # Pydantic settings
│   │   ├── db.py               # SQLite connection manager
│   │   └── types.py            # Shared types, enums
│   ├── steps/
│   │   ├── extract.py          # DataExtractStep
│   │   ├── transform.py        # FeatureTransformStep
│   │   ├── split.py            # TemporalSplitStep
│   │   ├── train.py            # ModelTrainStep
│   │   ├── validate.py         # ValidationStep
│   │   ├── persist.py          # ModelPersistStep
│   │   └── compare.py          # ModelCompareStep
│   ├── models/
│   │   ├── base.py             # SubmodelSpec ABC
│   │   └── swing_decision.py   # SwingDecisionSpec
│   ├── validation/
│   │   ├── metrics.py          # Metric functions
│   │   ├── calibration.py      # Calibration analysis
│   │   ├── thresholds.py       # Threshold definitions
│   │   └── report.py           # ValidationReport dataclass
│   └── viz/
│       ├── calibration_plot.py
│       ├── feature_importance.py
│       ├── roc_curve.py
│       └── swing_decision_viz.py  # Zone heatmaps, count breakdowns
└── tests/
    ├── conftest.py             # Tiny in-memory SQLite fixture
    ├── unit/                   # test_pipeline, test_artifact, test_extract, etc.
    └── regression/
        ├── test_swing_decision_performance.py
        └── thresholds.yaml
```

---

## Pipeline Architecture

### Core Abstractions

**`PipelineStep` (ABC):** Each step declares `input_artifact_keys()` and `output_artifact_keys()`. The `run(context)` method is idempotent — it checks if outputs exist and are fresher than inputs, skips if so (unless `--force`). Returns a `StepResult` with status (SUCCESS/SKIPPED/FAILED).

**`Pipeline`:** Validates the DAG at construction (every step's inputs must be satisfied by prior outputs or the DB). Runs steps sequentially, stops on failure.

**`SubmodelSpec` (ABC):** Extension point for new submodels. Defines extraction SQL, feature columns, transform function, metric thresholds. Adding a submodel = adding one new class.

**`ArtifactStore`:** Manages `artifacts/` directory. Parquet for DataFrames, joblib for models, JSON for reports. Tracks freshness timestamps.

### Data Flow

```
SQLite DB → [Extract] → raw.parquet → [Transform] → features.parquet
→ [Split] → train.parquet + test.parquet + manifest.json
→ [Train] → fitted models → [Persist] → .joblib files
→ [Validate] → ValidationReport JSON + plots
→ [Compare] → comparison report (optional)
```

---

## CLI Interface

```
hitplus run       --model swing_decision [--force] [--until STEP]
hitplus extract   --model swing_decision
hitplus train     --model swing_decision
hitplus validate  --model swing_decision [--plot]
hitplus compare   --baseline logistic_v1 --candidate lgbm_v1 --model swing_decision
hitplus inspect   --model swing_decision
hitplus viz       --model swing_decision --type calibration|importance|roc|zone_heatmap
```

Exit codes: 0=success, 1=validation failure, 2=error.
All commands accept `--full` to use the full dataset instead of dev mode (2023 only).

---

## Validation Framework (Critical Priority)

### Metrics & Thresholds

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| AUC-ROC | >= 0.82 | Discrimination |
| AUC-PR | >= 0.80 | Precision-recall (balanced classes) |
| Brier Score | <= 0.18 | Calibration quality |
| Log Loss | <= 0.55 | Probabilistic accuracy |
| ECE | <= 0.03 | Expected calibration error |
| Accuracy | >= 0.74 | Sanity check only |

Thresholds stored in `tests/regression/thresholds.yaml` and enforced in both validation step (warning) and regression tests (hard fail).

### Calibration Analysis (4 views)

1. **Overall** — predicted P(swing) vs observed, 10 and 20 bins
2. **By count** — 12 curves (each balls/strikes combo)
3. **By zone** — in-zone vs out-of-zone
4. **By pitch type** — top 6 pitch types

### ValidationReport Dataclass

Contains: model name, submodel, timestamps, date ranges, sample sizes, all metrics, thresholds, pass/fail status, calibration bins, feature importances. Serialized to JSON.

### Model Comparison

`ModelCompareStep` runs baseline and candidate on same test set, side-by-side metrics. New model only becomes baseline if it meets or exceeds all thresholds.

---

## Visualizations (7 plots)

All plots use `plt.style.use('ggplot')` for consistent styling.

1. **Calibration curve** — predicted vs observed P(swing), 20 bins, confidence bands
2. **Calibration by count** — 3x4 grid showing model knows 2-strike protect behavior
3. **ROC curve** — with AUC annotated
4. **Feature importance** — SHAP or permutation importance bar chart
5. **Zone heatmap** — P(swing) over plateX x plateZ with strike zone overlay, per count
6. **Zone heatmap: model vs actual** — side-by-side predicted vs observed
7. **Pitch type breakdown** — predicted vs observed swing rate by pitch type

---

## Step-by-Step Milestones

### Milestone 1: Project Skeleton & Core Abstractions
**Deliver:** `uv run hitplus --help` works. Core ABCs exist. `make lint` passes. Empty test suite passes.
**Files:** `pyproject.toml`, `Makefile`, `src/hitplus/{__init__,cli}.py`, `src/hitplus/core/{__init__,pipeline,artifact,config,db,types}.py`, `tests/conftest.py`
**Acceptance:** CLI prints help. Black/ruff pass. `pytest` collects 0 tests, exits 0.
**Fence:** No data access, no model code, no feature engineering.

### Milestone 2: ArtifactStore & Database Layer
**Deliver:** `ArtifactStore` can put/get parquet and JSON. `DatabaseConnection` returns DataFrames. Unit tests pass.
**Files:** Complete `artifact.py`, `db.py`. `tests/unit/test_artifact.py`.
**Acceptance:** Unit tests verify put/get/exists/freshness. DB returns correct DataFrame from test fixture.
**Fence:** No pipeline steps yet. Infrastructure only.

### Milestone 3: Pipeline Framework (Extract → Transform → Split)
**Deliver:** `hitplus run --model swing_decision --until split` produces train/test parquet files. Temporal split verified. Re-run is idempotent (steps skip).
**Files:** Complete `pipeline.py`, `steps/{extract,transform,split}.py`, `models/{base,swing_decision}.py`. Unit tests for each.
**Acceptance:** Parquet files exist with correct schemas. Split manifest shows correct date boundaries. No data leakage. Re-run shows SKIPPED status.
**Fence:** No training, no validation. Data pipeline only.

### Milestone 4: Model Training & Persistence
**Deliver:** `hitplus run --model swing_decision --until persist` fits LogisticRegression + LGBMClassifier, serializes to joblib.
**Files:** `steps/{train,persist}.py`, `tests/unit/test_train.py`.
**Acceptance:** Models load from disk. Predictions are in [0,1]. Re-run skips if models exist.
**Fence:** No validation yet. Confirm fit and serialize only.

### Milestone 5: Validation Framework
**Deliver:** `hitplus validate --model swing_decision` produces ValidationReport JSON. All metrics computed. Regression tests pass.
**Files:** `validation/{metrics,calibration,thresholds,report}.py`, `steps/validate.py`, `tests/unit/{test_validate,test_metrics,test_calibration}.py`, `tests/regression/{test_swing_decision_performance.py,thresholds.yaml}`.
**Acceptance:** All 6 metrics meet thresholds. Calibration bins computed for all 4 views. Report JSON is complete. `pytest -m regression` passes.
**Fence:** No plots yet. Numbers only.

### Milestone 6: Visualizations
**Deliver:** `hitplus viz --model swing_decision` generates all 7 plot types as PNGs.
**Files:** `viz/{calibration_plot,feature_importance,roc_curve,swing_decision_viz}.py`.
**Acceptance:** PNG files created in `artifacts/plots/swing_decision/`. Visual inspection confirms correctness.
**Fence:** Plots are additive. No changes to validation logic.

### Milestone 7: Model Comparison & CLI Polish
**Deliver:** `hitplus compare` works. `hitplus inspect` shows artifact inventory. All CLI commands have help text.
**Files:** `steps/compare.py`, updated `cli.py`.
**Acceptance:** Comparison report generated. All CLI subcommands work end-to-end.
**Fence:** Framework complete for first submodel. Ready for second.

---

## Dependencies

```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "polars>=1.0", "numpy>=1.25", "scikit-learn>=1.4",
    "lightgbm>=4.0", "shap>=0.44", "click>=8.1",
    "pydantic>=2.5", "pydantic-settings>=2.1",
    "joblib>=1.3",
    "matplotlib>=3.8", "seaborn>=0.13",
]
[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "black>=24.0", "ruff>=0.2"]
```

---

## Verification

After each milestone, run:
1. `make lint` — black + ruff pass
2. `make test` — unit tests pass
3. After M5+: `make test-regression` — performance thresholds met
4. After M6+: Visual inspection of plots in `artifacts/plots/`
5. After M7: Full end-to-end: `hitplus run --model swing_decision` completes all steps
