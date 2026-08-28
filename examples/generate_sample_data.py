"""Generate the sample attention arrays used by the README examples.

The README's stability and evolution examples originally loaded
`attention_matrices_5_runs_test_sample.npy` and `attention_matrices_epochs.npy`,
neither of which was ever committed, so neither example could be run.

Rather than commit binary fixtures, this script synthesises arrays with the
same shapes and broadly realistic structure: rows sum to one, attention is
biased towards the diagonal and towards [CLS], and successive runs/epochs
differ by a small amount of noise so the stability and evolution views have
something to show.

The output is illustrative, not real model output. To use genuine attention,
replace this with your own capture -- the shapes are what matter:

    (runs_or_epochs, layers, heads, seq_len, seq_len)

Usage:
    python examples/generate_sample_data.py
"""
import argparse
import pathlib

import numpy as np

RUNS = 5
LAYERS = 12
HEADS = 12
SEQ_LEN = 64


def _attention_like(rng, seq_len, sharpness):
    """One row-stochastic seq_len x seq_len matrix with diagonal and [CLS] bias."""
    positions = np.arange(seq_len)
    distance = np.abs(positions[:, None] - positions[None, :])
    logits = -sharpness * distance / seq_len
    logits[:, 0] += 1.0  # every token attends somewhat to [CLS]
    logits += rng.normal(0.0, 0.25, size=logits.shape)
    logits -= logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=-1, keepdims=True)


def build(rng, drift):
    """Stack of (RUNS, LAYERS, HEADS, SEQ_LEN, SEQ_LEN) float32 attention.

    `drift` controls how much each successive run/epoch departs from the
    first, which is what the stability and evolution views visualise.
    """
    base = np.stack(
        [
            [_attention_like(rng, SEQ_LEN, sharpness=2.0 + layer * 0.3) for _ in range(HEADS)]
            for layer in range(LAYERS)
        ]
    )
    out = np.empty((RUNS, LAYERS, HEADS, SEQ_LEN, SEQ_LEN), dtype=np.float32)
    for run in range(RUNS):
        noise = rng.normal(0.0, drift * run, size=base.shape) if run else 0.0
        perturbed = np.clip(base + noise, 1e-8, None)
        out[run] = (perturbed / perturbed.sum(axis=-1, keepdims=True)).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("."))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    targets = {
        # Five independent training runs: differences are noise, so the
        # stability view should show mostly-consistent attention.
        "attention_matrices_5_runs_test_sample.npy": build(rng, drift=0.004),
        # Five successive epochs of one run: attention drifts steadily, so
        # the sparklines have a visible trend.
        "attention_matrices_epochs.npy": build(rng, drift=0.010),
    }

    for name, array in targets.items():
        path = args.out_dir / name
        np.save(path, array)
        print(f"wrote {path}  shape={array.shape}  dtype={array.dtype}")


if __name__ == "__main__":
    main()
