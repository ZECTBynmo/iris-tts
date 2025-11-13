"""Data processing utilities for TTS."""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Optional
import jax.numpy as jnp


def load_audio(audio_path: str, sample_rate: int = 22050) -> np.ndarray:
    """
    Load audio file.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Audio waveform
    """
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    return audio


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
) -> np.ndarray:
    """
    Compute mel-spectrogram from audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Mel-spectrogram [n_mels, time]
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0,  # Use magnitude
    )
    
    # Convert to log scale
    mel_spec = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))
    
    return mel_spec


def normalize_mel_spectrogram(mel_spec: np.ndarray, 
                              mean: Optional[float] = None,
                              std: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """
    Normalize mel-spectrogram.
    
    Args:
        mel_spec: Mel-spectrogram
        mean: Pre-computed mean (if None, compute from data)
        std: Pre-computed std (if None, compute from data)
        
    Returns:
        Normalized mel-spectrogram, mean, std
    """
    if mean is None:
        mean = np.mean(mel_spec)
    if std is None:
        std = np.std(mel_spec)
    
    normalized = (mel_spec - mean) / (std + 1e-8)
    
    return normalized, mean, std
