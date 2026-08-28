"""Gradio app: interactive attention visualisation with IzzyViz.

Deployable as a Hugging Face Space. Type a sentence, pick a layer and head,
and get every IzzyViz view of the resulting attention.

Local:
    pip install -e ".[examples]" gradio
    python demo/app.py
"""
import tempfile

import gradio as gr
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

# Imported after matplotlib.use() on purpose: the backend must be selected
# before izzyviz pulls in pyplot, or the Space tries to open a display.
from izzyviz import (
    check_stability_heatmap_with_gradient_color,
    compare_two_attentions_with_circles,
    visualize_attention_matrix,
    visualize_attention_with_detected_regions,
)

MODEL_NAME = "distilbert-base-uncased"
MAX_TOKENS = 32  # keep figures legible and the Space responsive

_tokenizer = None
_model = None


def _load():
    """Load once, lazily -- a Space cold-starts faster this way."""
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
        _model.eval()
    return _tokenizer, _model


def _attention(text):
    tokenizer, model = _load()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        out = model(**encoded, output_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    return out.attentions, tokens


def _tmp_pdf():
    return tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name


def _simulate_runs(matrix, n=5, drift=0.015):
    """Synthetic repeat runs so the stability view has something to show.

    Capturing real repeated runs means retraining the model, which a Space
    cannot do. Clearly synthetic; the view is what is being demonstrated.
    """
    rng = np.random.default_rng(0)
    base = matrix.numpy()
    runs = [np.clip(base + rng.normal(0, drift * i, base.shape), 1e-8, None) for i in range(n)]
    runs = [r / r.sum(axis=-1, keepdims=True) for r in runs]
    return torch.tensor(np.stack(runs), dtype=torch.float32)


def visualize(text, layer, head, view, contrast_text):
    if not text.strip():
        return None, "Enter a sentence."

    attentions, tokens = _attention(text)
    layer = min(int(layer), len(attentions) - 1)
    head = min(int(head), attentions[0].shape[1] - 1)
    matrix = attentions[layer][0, head]
    path = _tmp_pdf()
    info = f"{len(tokens)} tokens · layer {layer} · head {head}\n{' '.join(tokens)}"

    if view == "Attention matrix":
        visualize_attention_matrix(matrix, x_labels=tokens, y_labels=tokens,
                                   title=f"layer {layer}, head {head}", save_path=path)
    elif view == "Detected regions":
        visualize_attention_with_detected_regions(
            matrix, source_tokens=tokens, target_tokens=tokens, save_path=path)
    elif view == "Compare two sentences":
        if not contrast_text.strip():
            return None, "Enter a second sentence to compare against."
        other_attn, other_tokens = _attention(contrast_text)
        if len(other_tokens) != len(tokens):
            return None, (f"Both sentences must tokenise to the same length "
                          f"({len(tokens)} vs {len(other_tokens)}).")
        compare_two_attentions_with_circles(
            matrix, other_attn[layer][0, head], tokens=tokens, save_path=path)
        info += f"\nvs: {' '.join(other_tokens)}"
    elif view == "Stability across runs":
        check_stability_heatmap_with_gradient_color(
            _simulate_runs(matrix), x_labels=tokens, y_labels=tokens, save_path=path)
        info += "\n(repeat runs are simulated -- see the note above)"
    else:
        return None, f"Unknown view: {view}"

    return path, info


with gr.Blocks(title="IzzyViz — transformer attention visualisation") as app:
    gr.Markdown(
        f"""# IzzyViz

Visualise attention in **{MODEL_NAME}**. Enter a sentence, pick a layer and
head, and choose a view. Output is a publication-quality PDF.

[Source](https://github.com/WING-NUS/IzzyViz) · WING, NUS School of Computing

*Stability uses simulated repeat runs — capturing real ones needs retraining.*
"""
    )
    with gr.Row():
        with gr.Column(scale=2):
            text = gr.Textbox(
                label="Sentence",
                value="The cat sat on the mat because it was tired.",
            )
            view = gr.Radio(
                [
                    "Attention matrix",
                    "Detected regions",
                    "Compare two sentences",
                    "Stability across runs",
                ],
                value="Attention matrix", label="View",
            )
            contrast = gr.Textbox(
                label="Second sentence (comparison view only — must tokenise to the same length)",
                value="The cat sat on the mat because it was comfortable.",
            )
            with gr.Row():
                layer = gr.Slider(0, 5, value=4, step=1, label="Layer")
                head = gr.Slider(0, 11, value=3, step=1, label="Head")
            go = gr.Button("Visualise", variant="primary")
        with gr.Column(scale=3):
            out_file = gr.File(label="Figure (PDF)")
            out_info = gr.Textbox(label="Tokens", lines=4)

    go.click(visualize, [text, layer, head, view, contrast], [out_file, out_info])

if __name__ == "__main__":
    app.launch()
