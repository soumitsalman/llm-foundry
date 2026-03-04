# llm-foundry

Utilities for dataset generation and fine-tuning compact language models so they reliably perform focused tasks with the quality and output-format consistency of larger models.

This repository provides tooling to create JSON-labeled training data from news, blogs, articles, and social media posts, and includes training scripts for both decoder-only models and text-to-text models.

## Key Features
- Dataset generation utilities for converting source text to compact, markdown-truncated inputs and JSON outputs.
- Fine-tuning scripts for decoder models (e.g., Hugging Face / SmolLM2 family) and text-to-text models (e.g., `allenai/led-base-*`).
- Emphasis on content fidelity, format reliability (JSON outputs), and preserving author tonality.
- Practical training guidance and hardware tips for medium-sized GPUs.

## What this repo contains
- `datagen.py`, `datagen_prompts.py`, `datasetgen.py` — data creation and labeling helpers.
- `datacleaning.py`, `converter.py` — preprocessing and markdown conversion utilities.
- `training_causal.py` — fine-tuning entrypoint for decoder-only models (SmolLM2-style, causal LM training).
- `training_s2s.py` — fine-tuning entrypoint for sequence-to-sequence / text-to-text models (LED, etc.).
- `download_models.py`, `utils.py` — helper scripts.

## Supported training workflows
- Decoder-only (causal) models: use `training_causal.py` to fine-tune models like [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) etc.
- Text-to-text (encoder-decoder) models: use `training_s2s.py` for models like led-base-16384[https://huggingface.co/allenai/led-base-16384] and other seq2seq architectures.

Both scripts are intended to be configurable via command-line arguments and compatible training frameworks; check the script headers for available flags.

## Quickstart
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare or generate a dataset using the provided generators (see `datagen.py` and `datasetgen.py`). Outputs are JSON rows with fields like `title`, `summary`, `highlights`, `names`, and `domains`.

3. Train a decoder model:

```bash
python training_causal.py --config configs/causal.yaml
```

4. Train a text-to-text model:

```bash
python training_s2s.py --config configs/s2s.yaml
```

## Practical notes & tips
- Keep per-device batch sizes and accumulation conservative to avoid OOM: prefer `per_device_train_batch_size` <= 96 and `gradient_accumulation_steps` <= 64 depending on GPU memory.
- Avoid loading extremely large datasets in a single run — recommended working set <= 400k rows; split and iterate for larger corpora.
- `os.cpu_count()` may report host physical CPU count which can exceed container limits; tune data loader workers manually.
- If a dataset contains a `labels` column, remove or rename it before passing to certain trainers (e.g., `dataset.remove_columns(["labels"])`).

## Output format
Training targets in this project prefer structured JSON responses. Typical label fields include:
- `title`, `summary`, `highlights`, `names`, `domains`

The generation tooling emphasizes producing concise, actionable summaries and preserving the original voice when requested.

## License
See the `LICENSE` file for license terms.

## Contact
For questions or contributions, open an issue or submit a PR.