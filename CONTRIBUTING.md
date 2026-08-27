# Contributing to KINO

KINO is a scientific-inference codebase: correctness of the physics and
reproducibility matter more than feature velocity. Please read `CLAUDE.md`
before contributing — it documents the unit system, coordinate/sign
conventions, and repository layout that every change must respect.

## Workflow

1. Check `README.md` section 4 (task list / roadmap) for the current phase
   and open items. Prefer picking up an unchecked item over starting
   unplanned work.
2. One phase item per PR where possible. Small, reviewable changes are
   easier to check for physics correctness than large ones.
3. Every PR should include, alongside the code:
   - tests (see below),
   - a `CHANGELOG.md` entry under `[Unreleased]`,
   - the relevant roadmap checkbox ticked in `README.md`.
4. If a change touches a sign, angle, or ellipticity/shear convention,
   update `CLAUDE.md`'s Conventions section, the affected docstrings, and
   the tests together — conventions must never drift silently.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a PR

```bash
ruff check .
ruff format --check .
pytest -q
mypy kino   # if/when typing is enabled for the touched module
```

All of these run in CI; a PR won't merge if they fail.

## Tests

- Prefer tests of invariants and limits (symmetries, round-trips,
  conservation laws, face-on/edge-on limits) over hard-coded numeric
  snapshots — see `CLAUDE.md` principle 4.
- Mirror the `kino/` package layout under `tests/`.
- Use a seeded `numpy.random.Generator` passed explicitly; never rely on
  global `np.random` state.

## Commit / PR style

- Commit messages: short imperative summary line, body explains *why* when
  not obvious.
- PR description: what changed, why, and which roadmap item(s) it closes.

## Reporting bugs / proposing features

Use the issue templates under `.github/ISSUE_TEMPLATE/`. For anything that
touches a physical convention, include a minimal reproducible example and,
if possible, the expected vs. actual numeric result.
