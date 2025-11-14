# HiFiGAN Pre-trained Vocoder Setup

This document describes how to use the pre-trained HiFiGAN vocoder in the Iris TTS pipeline.

## Quick Start

The pre-trained HiFiGAN model from speechbrain is already downloaded at:
```
models/hifigan/models--speechbrain--tts-hifigan-ljspeech/snapshots/17fbdc3aae35b81e1554111fa54eab5f2b70cedb/generator.ckpt
```

### 1. Dependencies

PyTorch and torchaudio have been added to `pyproject.toml`. Make sure they're installed:

```bash
uv sync
```

### 2. Test the Vocoder

```bash
uv run python test_hifigan_integration.py
```

This will:
- Load the pre-trained model
- Run inference with a dummy mel-spectrogram
- Verify everything works correctly

### 3. Use in Synthesis Pipeline

```bash
uv run python scripts/synthesize.py \
  --text "Your text here" \
  --vocoder hifigan \
  --vocoder_entry iris.hifigan_pretrained:infer_hifigan
```

Or use the example script:
```bash
bash example_hifigan_synthesis.sh
```

## Architecture

### Files Added

1. **`src/iris/hifigan_pretrained.py`**: PyTorch HiFiGAN implementation
   - `HiFiGANModel`: PyTorch model architecture
   - `HiFiGANGenerator`: Wrapper class for checkpoint loading
   - `infer_hifigan()`: Main inference function for use in pipelines
   - `get_pretrained_hifigan()`: Singleton pattern for model loading

2. **`test_hifigan_integration.py`**: Integration test script

3. **`example_hifigan_synthesis.sh`**: Example usage script

### Integration Points

The `scripts/synthesize.py` script already has vocoder support via the `--vocoder_entry` flag. The format is:
```
--vocoder_entry module:function
```

Where the function signature should be:
```python
def function(mel: np.ndarray, sample_rate: int, hop_length: int) -> np.ndarray:
    """
    Args:
        mel: [batch, n_mels, time] or [n_mels, time]
    Returns:
        audio: [samples] waveform
    """
```

## Usage in Code

### Option 1: Direct Inference (Recommended)

```python
from iris.hifigan_pretrained import infer_hifigan
import numpy as np

# Your mel-spectrogram from the TTS model
mel = np.random.randn(80, 100)  # Example: [n_mels, time]

# Generate audio
audio = infer_hifigan(mel)  # Returns [samples]
```

### Option 2: Get Vocoder Instance

```python
from iris.hifigan_pretrained import get_pretrained_hifigan

# Load model once
vocoder = get_pretrained_hifigan()

# Use for multiple inferences
audio1 = vocoder(mel1)
audio2 = vocoder(mel2)
```

### Option 3: Custom Checkpoint Path

```python
from iris.hifigan_pretrained import HiFiGANGenerator

vocoder = HiFiGANGenerator("/path/to/custom/generator.ckpt")
audio = vocoder(mel)
```

## Technical Details

### Model Architecture
- **Upsample rates**: [8, 8, 2, 2] → Total upsampling factor of 256
- **Resblock kernels**: [3, 7, 11]
- **Input**: 80-channel mel-spectrogram
- **Output**: Raw audio waveform at 22050 Hz

### Mel-spectrogram Format
The model expects log-magnitude mel-spectrograms with:
- 80 mel channels
- Sample rate: 22050 Hz
- Hop length: 256 samples
- n_fft: 1024

### Device Support
- Automatically uses CUDA if available
- Falls back to CPU otherwise

## Comparison: HiFiGAN vs Griffin-Lim

| Aspect | HiFiGAN | Griffin-Lim |
|--------|---------|-------------|
| Quality | High (neural vocoder) | Moderate (iterative algorithm) |
| Speed | Fast (single forward pass) | Slower (60 iterations) |
| Training | Pre-trained available | Not required |
| Dependencies | PyTorch | librosa only |

## Troubleshooting

### ImportError: No module named 'torch'
Run `uv sync` to install PyTorch and torchaudio.

### Checkpoint not found
Verify the checkpoint exists at:
```
models/hifigan/models--speechbrain--tts-hifigan-ljspeech/snapshots/.../generator.ckpt
```

### Shape mismatch errors
Ensure your mel-spectrogram has 80 channels (n_mels=80) and is in log scale.

## Future Enhancements

Possible improvements:
1. Support for other HiFiGAN variants (V2, V3)
2. Fine-tuning on custom datasets
3. ONNX export for production deployment
4. JAX conversion for full Keras/JAX pipeline

