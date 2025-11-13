# Iris - Text-to-Speech Model

A Text-to-Speech (TTS) model built with Keras and JAX.

## Setup (Conda + uv)

We use Conda for compiled deps (pynini/OpenFst/MFA) and uv for everything else.

1) Install uv (if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Create and activate the Conda env (macOS arm64)
```bash
conda create -n iris-tts -c conda-forge python=3.10 -y
conda activate iris-tts
```

3) Install compiled dependencies from conda-forge
```bash
conda install -y -c conda-forge pynini=2.1.6.post1 openfst thrax montreal-forced-aligner
```

4) Create uv’s venv using the Conda interpreter and inherit Conda packages
```bash
uv venv --python "$(which python)" --system-site-packages
source .venv/bin/activate
```

5) Sync Python dependencies with uv
```bash
uv sync
```

6) Verify
```bash
python -c "import _pynini, pynini; import importlib.metadata as md; print('pynini', md.version('pynini'))"
python -c "from nemo_text_processing.text_normalization.normalize import Normalizer; print(Normalizer(input_case='cased', lang='en').normalize('I have $12.50'))"
```

Notes:
- Keep the `iris-tts` Conda env active when working with this repo, then activate `.venv`.
- Avoid Python 3.13 for now; `pynini` wheels are not available and source builds are brittle on macOS arm64.

## Project Structure

```
iris/
├── src/
│   └── iris/
│       ├── __init__.py
│       ├── model.py
│       └── data.py
├── pyproject.toml
└── README.md
```

## Usage

### Text Processing (CMUDict + g2p_en)

Convert text to phonemes for TTS training:

```bash
# Run the text processing demo
uv run python demo_text_processing.py
```

**Usage in code:**

```python
from iris.text import create_text_processor

# Initialize processor
processor = create_text_processor(use_g2p=True)

# Convert text to phonemes
text = "Hello world, this is a test"
phonemes = processor.text_to_phonemes(text)
print(phonemes)  # "HH EH L OW W ER L D DH IH S IH Z AH T EH S T"

# Create phoneme vocabulary
texts = ["training text 1", "training text 2", ...]
phoneme_to_id, id_to_phoneme = processor.create_phoneme_mapping(texts)

# Convert to sequence of IDs
sequence = processor.text_to_sequence(text, phoneme_to_id)
```

### Basic TTS Model

```python
from iris.model import TTSPipeline

# Initialize and use the TTS model
pipeline = TTSPipeline()
audio = pipeline.synthesize("Hello, world!")
```

### Forced Alignment with Montreal Forced Aligner

The project includes utilities for Montreal Forced Aligner (MFA) for phoneme-level alignment of the LJSpeech dataset.

#### Setup (one-time)

```bash
# Already covered above: MFA is installed in the Conda env (iris-tts).
# Ensure both envs are active when you work:
conda activate iris-tts
source .venv/bin/activate
```

#### Quick Start

```bash
# Make sure the Conda env and uv venv are active, then:
uv run python align_ljspeech.py
```

This will:
1. Extract text from metadata.csv
2. Download MFA models (first run only, ~2-3GB)
3. Align audio to text phoneme-by-phoneme
4. Save alignments as TextGrid files

#### Using Alignments in Your Code

```python
from iris.alignment import MFAAligner

aligner = MFAAligner()
alignments = aligner.load_alignments("data/ljspeech_alignments")

# Access phoneme-level timing info
for filename, phones in alignments.items():
    for phone in phones:
        print(f"{phone['phone']}: {phone['start']:.3f}s - {phone['end']:.3f}s")
```

### HiFiGAN Vocoder (Keras/JAX)

HiFiGAN is a high-quality neural vocoder that converts mel-spectrograms to audio waveforms. Our implementation uses **Keras with JAX backend** - no PyTorch needed!

#### Quick Demo

```bash
# Run the HiFiGAN architecture demo
uv run python demo_vocoder.py
```

This will:
1. Initialize HiFiGAN model in Keras/JAX
2. Extract mel-spectrogram from a sample LJSpeech audio
3. Generate audio using HiFiGAN architecture
4. Show model summary and usage examples

**Note:** The demo uses random weights. For production use, you need to train the model on your dataset.

#### Using HiFiGAN in Your Code

```python
from iris.vocoder import create_vocoder
import numpy as np
import soundfile as sf

# Initialize vocoder
vocoder = create_vocoder(weights_path="models/hifigan_weights.keras")

# Generate audio from mel-spectrogram
mel = your_tts_model.synthesize("Hello")  # Shape: [80, time]
audio = vocoder.infer(mel)                # Returns: [samples]

# Save audio
sf.write("output.wav", audio, 22050)
```

#### Training HiFiGAN

To train HiFiGAN on your dataset:
1. Prepare paired (mel-spectrogram, audio) data from LJSpeech
2. Train the generator model
3. Save weights: `vocoder.save_weights("hifigan_weights.keras")`
4. Use trained weights for inference

## Development

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black src/
```

Lint code:
```bash
uv run ruff check src/
```

