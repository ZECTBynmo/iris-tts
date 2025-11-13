#!/usr/bin/env python3
"""
HiFiGAN Vocoder Demo using Keras/JAX.

This demonstrates the HiFiGAN architecture implemented in Keras.
Note: The model shown here has random weights. For actual use, you'll need
to train it on your dataset or convert pre-trained weights.

Usage:
    python demo_vocoder.py
"""

import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import keras

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_mel_spectrogram(
    audio_path: str,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: int = 0,
    fmax: int = 8000,
    sample_rate: int = 22050,
) -> np.ndarray:
    """
    Extract mel-spectrogram from audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        mel: Mel-spectrogram [mel_channels, time]
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    
    # Convert to log scale
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    
    return mel


def main():
    logger.info("=" * 70)
    logger.info("HiFiGAN Vocoder Demo (Keras/JAX)")
    logger.info("=" * 70)
    
    # Set Keras backend to JAX
    logger.info(f"\nUsing Keras backend: {keras.backend.backend()}")
    
    # Step 1: Initialize vocoder
    logger.info("\n[1/4] Initializing HiFiGAN vocoder...")
    
    try:
        from iris.vocoder import create_vocoder
        
        # Create vocoder (with random weights - needs training for actual use)
        vocoder = create_vocoder()
        
    except Exception as e:
        logger.error(f"Failed to initialize vocoder: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Step 2: Load sample audio and extract mel-spectrogram
    logger.info("\n[2/4] Processing sample audio...")
    
    sample_audio = Path("data/LJSpeech-1.1/wavs/LJ001-0001.wav")
    
    if not sample_audio.exists():
        logger.error(f"Sample audio not found: {sample_audio}")
        logger.info("Please ensure LJSpeech dataset is downloaded")
        logger.info("\nCreating synthetic mel-spectrogram for demo...")
        # Create synthetic mel for demo
        mel = np.random.randn(80, 100) * 0.5
    else:
        # Extract mel-spectrogram from real audio
        mel = extract_mel_spectrogram(str(sample_audio))
    
    logger.info(f"Mel-spectrogram shape: {mel.shape}")
    
    # Step 3: Generate audio with vocoder
    logger.info("\n[3/4] Generating audio with HiFiGAN...")
    logger.info("Note: Model has random weights, so output will be noise.")
    logger.info("For real use, you need to train the model or load pre-trained weights.")
    
    try:
        generated_audio = vocoder.infer(mel)
        logger.info(f"Generated audio shape: {generated_audio.shape}")
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Step 4: Save output
    logger.info("\n[4/4] Saving generated audio...")
    output_dir = Path("outputs/vocoder")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generated_audio_random.wav"
    
    # Normalize audio to prevent clipping
    audio_normalized = generated_audio / (np.abs(generated_audio).max() + 1e-8) * 0.95
    
    sf.write(output_path, audio_normalized, 22050)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ Demo complete!")
    logger.info("=" * 70)
    logger.info(f"\nGenerated audio saved to: {output_path}")
    logger.info(f"Audio length: {len(generated_audio) / 22050:.2f} seconds")
    logger.info("\nIMPORTANT: The output is noise because the model has random weights.")
    logger.info("To use HiFiGAN in production:")
    logger.info("1. Train the model on paired (mel, audio) data from LJSpeech")
    logger.info("2. Save weights: vocoder.save_weights('hifigan_weights.keras')")
    logger.info("3. Load weights: vocoder = create_vocoder('hifigan_weights.keras')")
    
    # Show usage example
    logger.info("\n" + "=" * 70)
    logger.info("Usage in your TTS pipeline:")
    logger.info("=" * 70)
    print("""
from iris.vocoder import create_vocoder
import numpy as np
import soundfile as sf

# Initialize vocoder with trained weights
vocoder = create_vocoder(weights_path="models/hifigan_weights.keras")

# Your TTS model generates mel-spectrogram
mel = your_tts_model.synthesize("Hello, world!")  # Shape: [80, time]

# Convert mel to audio
audio = vocoder.infer(mel)  # Returns: [samples]

# Save audio
sf.write("output.wav", audio, 22050)
""")
    
    # Show model architecture
    logger.info("\n" + "=" * 70)
    logger.info("Model Architecture:")
    logger.info("=" * 70)
    vocoder.model.summary()


if __name__ == "__main__":
    main()

