# Contributing to Pocket AI Guardian

Thanks for your interest in improving the project.

## Setup

The repository provides a conda environment (`environment.yml`, Python 3.12) and
a `requirements.txt`. CI runs on Python 3.11.

```bash
git clone https://github.com/youhanasheriff/pocket_ai.git
cd pocket_ai

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run only the unit tests you do not need the full dependency set. The same
minimal set that CI uses is enough:

```bash
pip install -r requirements-ci.txt
```

## Running the tests

```bash
python -m pytest tests/
```

The tests cover the validator, spatial, instructor and depth stages and do not
require a camera or model weights. Please keep it that way: new tests should not
depend on hardware, network access, or downloading models.

## Linting

CI runs `ruff check .` as an informational (non-blocking) step, because the
existing code has not been reformatted. Please avoid introducing new warnings in
code you touch, and do not mix large formatting-only changes into functional
pull requests.

```bash
ruff check .
```

## Pull requests

- Branch from `main` and open the pull request against `main`.
- Keep pull requests focused; describe what changed and why.
- Add or update tests in `tests/` for behavior changes.
- Make sure `python -m pytest tests/` passes locally before requesting review.
- Update `README.md` or `docs/` if you change user-facing behavior or configuration.
- Do not commit secrets, `.env` files, or large new model binaries.

## Reporting security issues

See [SECURITY.md](SECURITY.md).
