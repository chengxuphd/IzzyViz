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

1. Create a Space with the **Gradio** SDK.
2. Copy `app.py` and this `README.md` into it (the YAML header above is the
   Space config).
3. Add a `requirements.txt`:

   ```
   git+https://github.com/WING-NUS/IzzyViz.git
   transformers>=4.0.0
   torch
   gradio
   ```

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
