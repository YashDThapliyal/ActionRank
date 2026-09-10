# ActionRank

LLM-native ranking for agent tool/action selection: score every tool in a catalog with a single
prefill pass instead of generating a tool call token by token. See `spec.md`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

All scripts read `config.yaml`. Set `model.device` to `mps` (Apple Silicon), `cuda` (Colab) or `cpu`.
