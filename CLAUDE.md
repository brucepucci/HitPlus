# Hit+ ML Framework

## Project Overview

Pitch-by-pitch hitter outcome model inspired by Stuff+/PLV but from the batter's perspective. Multiple submodels (swing decision, called strike, contact quality) score each pitch individually; they will eventually combine into a single Hit+ metric.

**Repo:** https://github.com/brucepucci/HitPlus.git
**Data:** `data/mlb_stats.db` (~4.9GB SQLite, git-ignored) — 5.3M regular season pitches, 2018-2025

## Tech Stack

- **Python 3.11+** managed by **uv**
- **polars + SQL** for data transformations
- **scikit-learn** + **LightGBM** for modeling
- **Click** for CLI (`hitplus` entrypoint)
- **Pydantic** for config
- **pytest** for unit and regression tests
- **matplotlib/seaborn** for visualizations (ggplot style: `plt.style.use('ggplot')`)

## Style & Conventions

- **black** formatting (line length 88)
- **ruff** for linting
- OOP where it makes sense, Pythonic style preferred, don't force abstractions
- Simple models first — only add complexity for significant performance gains
- Dev mode by default (2024 only, random 70/30 split); `--full` for 2022-2024 temporal train / 2025 test

## Project Structure

- `src/hitplus/` — main package
  - `core/` — pipeline framework, artifact store, config, DB connection
  - `steps/` — pipeline steps (extract, transform, split, train, validate, persist, compare)
  - `models/` — submodel specs (swing_decision.py, etc.)
  - `validation/` — metrics, calibration, thresholds, reports
  - `viz/` — visualization generators
  - `cli.py` — Click CLI entrypoint
- `tests/unit/` — unit tests (run with `make test`)
- `tests/regression/` — performance threshold tests (run with `make test-regression`)
- `artifacts/` — git-ignored pipeline outputs (datasets, models, validation reports, plots)
- `data/` — git-ignored SQLite database

## Key Commands

```bash
make lint              # black + ruff
make test              # unit tests
make test-regression   # performance threshold tests
hitplus run --model swing_decision          # full pipeline (dev mode)
hitplus run --model swing_decision --full   # full pipeline (all training data)
```

## Database Notes

- Schema reference: `~/.claude/skills/mlb-stats-query/references/schema.md`
- `pitches.balls`/`strikes` = count BEFORE the pitch, not after
- Swing decision model: no exclusions. All call codes classified as swing or take (see PLAN.md for full table)
- Statcast fields (spinRate, extension, plateTime) are NULL pre-2015
- Always filter `g.gameType = 'R'` for regular season

## Architecture Principles

- Pipeline steps are **idempotent** — skip if outputs are fresher than inputs, re-run with `--force`
- Each step declares input/output artifact keys explicitly
- Adding a new submodel = writing one `SubmodelSpec` subclass
- Validation is the top priority — 6 metrics with hard thresholds, 4 calibration views, regression test gates
- Dev mode uses random 70/30 split on 2024; full mode uses temporal split (2022-2024 train / 2025 test) to prevent leakage
- 2025 season is the test set in full mode
