# Contributing to IzzyViz

Issues and pull requests are welcome, including on
[WING-NUS/IzzyViz](https://github.com/WING-NUS/IzzyViz) — that is the
maintained repository.

## Development setup

```bash
git clone https://github.com/WING-NUS/IzzyViz.git
cd IzzyViz
python -m venv .venv && source .venv/bin/activate

python -m pip install --upgrade pip   # older pips fail on the CPU torch index
# CPU-only torch keeps the install small; drop the index URL if you want CUDA.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest tests/                   # the whole suite
pytest tests/test_utility.py    # geometry layer only
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

If you add a public function, add it to `izzyviz/__init__.__all__` and call
it from at least one test in `tests/test_visualization.py`.
`TestPublicApi.test_every_export_is_called_by_a_test` fails if you do not.

## Demos

`examples/demo.py` runs DistilBERT and renders five views from real attention:
the plain matrix, comparison, stability, evolution and detected regions. It uses the public API only, so it doubles as an executable check
of the exported surface against genuine model output rather than the synthetic
matrices the test suite uses. Run it after any change to a visualisation
function.

`demo/app.py` is the Gradio version, deployable as a Hugging Face Space; see
`demo/README.md`.

## Notes for maintainers

- `izzyviz/my_seaborn.py` is a **BSD-3-Clause derivative of seaborn**, not
  original code, and not a drop-in for `import seaborn`. Read its module
  header before changing it. IzzyViz itself is MIT; see `NOTICE`.
- Several functions default `save_path` to a bare filename, so calling them
  without one writes a PDF into the current working directory. Pass an
  explicit path. (`visualize_attention_encoder_decoder` used to ignore an
  explicit path too; that is fixed and covered by `TestEncoderDecoder`.)
- The `numpy<2.0.0` pin in `setup.py` guards a real numpy-2 deprecation in
  `check_stability_heatmap_with_gradient_color` (`np.array()` on a torch
  tensor). Fix that call site before relaxing it.
