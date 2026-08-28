"""End-to-end demo: real attention from a real model, through every IzzyViz view.

Runs DistilBERT over one sentence and renders each of the library's
visualisations from the resulting attention tensors. Everything here uses the
public API only, so it doubles as an executable smoke test of the exported
surface against genuine model output rather than synthetic matrices.

Three of these views (comparison, evolution, region detection) raised
TypeError on every call before the gamma fix on this branch; they are included
deliberately so the demo exercises what was repaired.

Usage:
    pip install -e ".[examples]"
    python examples/demo.py --out-dir demo_output
"""
import argparse
import pathlib

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

# Imported after matplotlib.use() on purpose: the backend must be selected
# before izzyviz pulls in pyplot, or a headless run tries to open a display.
from izzyviz import (
    check_stability_heatmap_with_gradient_color,
    compare_two_attentions_with_circles,
    visualize_attention_evolution_sparklines,
    visualize_attention_matrix,
    visualize_attention_with_detected_regions,
)

MODEL = "distilbert-base-uncased"
SENTENCE = "The cat sat on the mat because it was tired."
CONTRAST = "The cat sat on the mat because it was comfortable."
LAYER, HEAD = 4, 3


def attention_for(model, tokenizer, text):
    """Return (attentions, tokens) for one sentence.

    attentions is a tuple of (batch, heads, seq, seq) tensors, one per layer.
    """
    encoded = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded, output_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    return output.attentions, tokens


def simulate_training_runs(attention, n_runs=5, drift=0.01, seed=0):
    """Perturb one attention matrix into a stack of plausible repeat runs.

    The stability and evolution views need several matrices for the same
    layer/head. Capturing those properly means retraining a model N times,
    which is out of scope for a demo, so we add small noise and renormalise.
    Clearly synthetic -- the point is to exercise the view, not to make a
    claim about training dynamics.
    """
    rng = np.random.default_rng(seed)
    base = attention.numpy()
    runs = []
    for run in range(n_runs):
        noisy = np.clip(base + rng.normal(0, drift * run, base.shape), 1e-8, None)
        runs.append(noisy / noisy.sum(axis=-1, keepdims=True))
    return torch.tensor(np.stack(runs), dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("demo_output"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--head", type=int, default=HEAD)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    print(f"loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, attn_implementation="eager")
    model.eval()

    attentions, tokens = attention_for(model, tokenizer, SENTENCE)
    n_layers, n_heads = len(attentions), attentions[0].shape[1]
    print(f"  {n_layers} layers x {n_heads} heads, "
          f"{len(tokens)} tokens: {' '.join(tokens)}")

    # Clamp rather than crash: --model accepts any HF encoder, and the
    # defaults are sized for DistilBERT (6 layers, 12 heads).
    layer_idx = min(args.layer, n_layers - 1)
    head_idx = min(args.head, n_heads - 1)
    if (layer_idx, head_idx) != (args.layer, args.head):
        print(f"  requested layer {args.layer}/head {args.head} is out of range "
              f"for this model; using layer {layer_idx}/head {head_idx}")

    head = attentions[layer_idx][0, head_idx]
    outputs = {}

    # 1. The basic view: one layer, one head.
    path = args.out_dir / "1_attention_matrix.pdf"
    visualize_attention_matrix(head, x_labels=tokens, y_labels=tokens,
                               title=f"layer {layer_idx}, head {head_idx}", save_path=str(path))
    outputs["attention matrix"] = path

    # 2. Region detection: where is this head actually looking?
    path = args.out_dir / "2_detected_regions.pdf"
    visualize_attention_with_detected_regions(
        head, source_tokens=tokens, target_tokens=tokens, save_path=str(path))
    outputs["detected regions"] = path

    # 3. Comparison: does "tired" vs "comfortable" change the pattern?
    contrast_attn, contrast_tokens = attention_for(model, tokenizer, CONTRAST)
    if len(contrast_tokens) == len(tokens):
        path = args.out_dir / "3_comparison.pdf"
        compare_two_attentions_with_circles(
            head, contrast_attn[layer_idx][0, head_idx], tokens=tokens, save_path=str(path))
        outputs["comparison"] = path

    # 4-5. Stability and evolution need repeated observations; see the
    #      docstring for why these are simulated rather than captured.
    runs = simulate_training_runs(head)
    path = args.out_dir / "4_stability.pdf"
    check_stability_heatmap_with_gradient_color(
        runs, x_labels=tokens, y_labels=tokens, save_path=str(path))
    outputs["stability"] = path

    epochs = simulate_training_runs(head, n_runs=5, drift=0.03, seed=1)
    over_time = epochs[:, None, None, :, :]  # (time, layer, head, seq, seq)
    path = args.out_dir / "5_evolution.pdf"
    visualize_attention_evolution_sparklines(
        over_time, tokens=tokens, layer=0, head=0, save_path=str(path))
    outputs["evolution"] = path

    print(f"\nwrote {len(outputs)} figures to {args.out_dir}/")
    for name, path in outputs.items():
        print(f"  {name:20s} {path.name}  ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
