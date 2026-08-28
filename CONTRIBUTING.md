# Contributing to IzzyViz

Issues and pull requests are welcome, including on
[WING-NUS/IzzyViz](https://github.com/WING-NUS/IzzyViz) — that is the
maintained repository.

## Development setup

```bash
git clone https://github.com/WING-NUS/IzzyViz.git
cd IzzyViz
python -m venv .venv && source .venv/bin/activate

# CPU-only torch keeps the install small; drop the index URL if you want CUDA.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest tests/          # all 38
pytest tests/test_utility.py    # geometry only; needs no torch or matplotlib
```

`tests/conftest.py` forces the matplotlib `Agg` backend, so the suite runs
headless. Tests must not require a network connection, a model download, or a
GPU — the integration tests drive the public API with small synthetic
attention matrices instead.

## What tests should assert

`tests/test_visualization.py` checks that each exported function runs and
writes a real PDF (verified by magic bytes). It deliberately does **not**
assert what the picture looks like: pixel comparisons are fragile across
matplotlib and font versions. If you add a golden-image test, mark it so it
can be skipped.

If you add a public function, add it to `izzyviz/__init__.__all__` and give
it at least one integration test — `TestPublicApi` will otherwise not see it.

## Notes for maintainers

- `izzyviz/my_seaborn.py` is a **BSD-3-Clause derivative of seaborn**, not
  original code, and not a drop-in for `import seaborn`. Read its module
  header before changing it. IzzyViz itself is MIT; see `NOTICE`.
- Several functions default `save_path` to a bare filename, so calling them
  writes a PDF into the current working directory. Pass an explicit path.
- The `numpy<2.0.0` pin in `setup.py` guards a real numpy-2 deprecation at
  `visualization.py:2256`. Fix that call site before relaxing it.
