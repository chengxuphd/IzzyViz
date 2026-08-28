---
title: IzzyViz
emoji: 🔍
colorFrom: purple
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# IzzyViz — transformer attention visualisation

Interactive demo of [WING-NUS/IzzyViz](https://github.com/WING-NUS/IzzyViz).
Enter a sentence, pick a layer and head in DistilBERT, and render any of the
library's views as a publication-quality PDF.

## Deploying to a Hugging Face Space

The three files in this directory are the complete Space: `app.py`, this
`README.md` (its YAML header is the Space config), and `requirements.txt`.

```bash
pip install huggingface_hub
huggingface-cli login                    # needs a WRITE token

huggingface-cli repo create izzyviz --type space --space_sdk gradio
git clone https://huggingface.co/spaces/chengxuphd/izzyviz /tmp/space
cp demo/{app.py,README.md,requirements.txt} /tmp/space/
cd /tmp/space && git add -A && git commit -m "IzzyViz demo" && git push
```

> **`requirements.txt` points at the `maintenance-pass` branch of the fork,
> not WING-NUS.** Three of the four views raise `TypeError` on WING-NUS
> `main`. Repoint it once the branch is merged.

First build takes a few minutes; DistilBERT (~250 MB) downloads on the first
request, so the initial visualisation is slower than later ones.

## Running locally

```bash
pip install -e ".[examples]" gradio
python demo/app.py
```

## Notes

- The stability view uses **simulated** repeat runs. Capturing real ones means
  retraining the model, which a Space cannot do; the view is what is being
  demonstrated, not a claim about training dynamics.
- Input is truncated to 32 tokens to keep figures legible and the Space
  responsive.
- The comparison view requires both sentences to tokenise to the same length.
