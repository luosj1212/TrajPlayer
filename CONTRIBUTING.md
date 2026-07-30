# Contributing

Thanks for helping improve TrajPlayer. Bug reports, format compatibility cases,
performance traces, documentation fixes, and focused code changes are welcome.

## Set Up

Use Python 3.10 or 3.11:

```bash
python -m venv .venv
```

Activate the environment, then install the project and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before Opening a Pull Request

```bash
python -m pytest -q
python -m compileall -q app.py trajplayer scripts
python scripts/check_release.py
```

Keep trajectory fixtures small and redistributable. Do not commit generated
`.tpdata` caches, benchmark stores, user trajectories, build directories, or
release archives.

## Performance Changes

For renderer, streaming, or playback changes, include the GPU, driver, operating
system, atom count, bond count, display mode, and benchmark command. Compare
visual output as well as timing; performance work must not silently lower image
quality.

## Scope

- Preserve contiguous float32 frame data and bounded-memory streaming.
- Keep trajectory I/O and expensive buffer preparation off the Qt UI thread.
- Avoid per-atom Python rendering loops.
- Add focused tests for behavior changes.
- Keep unrelated refactors out of the same pull request.

By contributing, you agree that your contribution is licensed under the MIT
License in this repository.
