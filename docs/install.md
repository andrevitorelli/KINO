# Install

KINO targets Python ≥ 3.10. Clone the repository and install in editable
mode:

```bash
git clone https://github.com/andrevitorelli/KINO.git
cd KINO
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e "."` alone pulls in the core dependencies (NumPy, SciPy,
Astropy, GalSim, `nautilus-sampler`, Matplotlib, PyYAML, h5py). The rest are
optional extras, installed as `pip install -e ".[extra]"` (or combined,
e.g. `".[dev,notebooks]"`):

| Extra | What it adds | When you need it |
|---|---|---|
| `dev` | `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`, plus `data` | Developing KINO itself; this is what CI installs. |
| `data` | `kinms` | Generating KinMS-based mock datacubes (`kino.io.kinms`). |
| `sky` | `skycalc-cli` | Rebuilding the offline ESO SkyCalc sky library (`scripts/build_skycalc_library.py`) — not needed to *use* the library, only to regenerate it. |
| `notebooks` | `jupyterlab`, `ipykernel` | Running the tutorial notebooks in `notebooks/`. |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building this documentation site locally. |

## Verify the install

```bash
pytest -q
```

should pass with no failures. If you installed the `dev` extra:

```bash
ruff check . && ruff format --check .
```

should also pass clean — both run in CI on every push and pull request.

## Building the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

serves this site at `http://127.0.0.1:8000` with live reload. `mkdocs build
--strict` is what CI runs before deploying — it fails on any broken
cross-reference or missing docstring, so run it locally before pushing
documentation changes.
