# Setup Instructions

This project uses Conda for compiled dependencies and uv for Python package management.

## 1) Install uv (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2) Create the Conda environment (macOS arm64)
```bash
conda create -n iris-tts -c conda-forge python=3.10 -y
conda activate iris-tts
```

## 3) Install compiled dependencies from conda-forge
```bash
conda install -y -c conda-forge pynini=2.1.6.post1 openfst thrax montreal-forced-aligner
```

## 4) Create uv’s virtualenv that reuses Conda packages
```bash
uv venv --python "$(which python)" --system-site-packages
source .venv/bin/activate
```

## 5) Sync project dependencies
```bash
uv sync
```

## Verification
```bash
# Check pynini
python -c "import _pynini, pynini; import importlib.metadata as md; print('pynini', md.version('pynini'))"

# Check NeMo text processing
python - <<'PY'
from nemo_text_processing.text_normalization.normalize import Normalizer
print(Normalizer(input_case='cased', lang='en').normalize('I have $12.50'))
PY

# Check MFA
mfa version
```

## Daily workflow
```bash
conda activate iris-tts
source .venv/bin/activate
uv sync   # when pyproject changes
uv run python demo_text_processing.py
```

## Troubleshooting
- ModuleNotFoundError: No module named '_pynini'
  - Ensure both `conda activate iris-tts` and `source .venv/bin/activate` are active.
  - The uv venv must be created with: `uv venv --python "$(which python)" --system-site-packages`.
  - Avoid Python 3.13; use Python 3.10 in the Conda env.
  - Clear `PYTHONPATH` if set: `unset PYTHONPATH`.

- uv installs into the wrong interpreter
  - Be explicit: `uv pip install --system --python "$(which python)" PACKAGE`.

- MFA not found
  - Make sure Conda env is active (`mfa` is installed into the Conda env).
