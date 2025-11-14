"""Pre-trained HiFiGAN vocoder using PyTorch models.

This module provides a bridge between the Keras/JAX TTS pipeline and
pre-trained PyTorch HiFiGAN vocoders, allowing you to use high-quality
pre-trained models without training from scratch.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List
import numpy as np

logger = logging.getLogger(__name__)

# Import torch at module level for proper class inheritance
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


def _ensure_torch():
    """Ensure PyTorch is available."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for pre-trained HiFiGAN. "
            "Install with: uv sync"
        )
    return torch, nn


class ResBlock(nn.Module if _TORCH_AVAILABLE else object):
    """HiFiGAN residual block with dilated convolutions."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[int] = [1, 3, 5]):
        super().__init__()
        
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        
        for dilation in dilations:
            self.convs1.append(
                nn.utils.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size,
                             dilation=dilation, padding=self._get_padding(kernel_size, dilation))
                )
            )
            self.convs2.append(
                nn.utils.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size,
                             padding=self._get_padding(kernel_size, 1))
                )
            )
    
    def _get_padding(self, kernel_size: int, dilation: int = 1):
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANModel(nn.Module if _TORCH_AVAILABLE else object):
    """PyTorch HiFiGAN Generator architecture."""
    
    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Pre-conv
        self.conv_pre = nn.utils.weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        )
        
        # Upsample layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # Post-conv
        self.conv_post = nn.utils.weight_norm(
            nn.Conv1d(ch, 1, 7, padding=3)
        )
    
    def forward(self, x):
        x = self.conv_pre(x)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            # Multi-receptive field fusion
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class HiFiGANGenerator:
    """Wrapper for pre-trained HiFiGAN generator from PyTorch checkpoint."""
    
    def __init__(self, checkpoint_path: Union[str, Path]):
        """
        Load pre-trained HiFiGAN generator.
        
        Args:
            checkpoint_path: Path to generator.ckpt file
        """
        torch, nn = _ensure_torch()
        
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logger.info(f"Loading HiFiGAN from: {self.checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        
        # The checkpoint should be a state dict (speechbrain format)
        if hasattr(checkpoint, 'eval'):
            # It's already a model object
            self.model = checkpoint
            logger.info("Loaded model directly from checkpoint")
        elif isinstance(checkpoint, dict):
            # Extract state dict if it's a nested dictionary
            if "generator" in checkpoint:
                state_dict = checkpoint["generator"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
            
            # Create model with matching architecture
            logger.info("Creating HiFiGAN model with standard architecture...")
            self.model = HiFiGANModel()
            
            # Load the weights
            try:
                self.model.load_state_dict(state_dict, strict=False)
                logger.info("✓ Loaded state dict successfully")
            except Exception as e:
                logger.error(f"Failed to load state dict: {e}")
                raise RuntimeError(
                    f"Could not load HiFiGAN checkpoint. "
                    f"The checkpoint format may not be compatible. "
                    f"Error: {e}"
                )
        else:
            raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"✓ HiFiGAN loaded successfully on device: {self.device}")
    
    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """
        Generate audio from mel-spectrogram.
        
        Args:
            mel: Mel-spectrogram as numpy array
                 Shape: [batch, n_mels, time] or [n_mels, time]
        
        Returns:
            audio: Generated audio waveform [batch, samples] or [samples]
        """
        torch, _ = _ensure_torch()
        
        # Handle different input shapes
        squeeze_batch = False
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]  # [n_mels, time] -> [1, n_mels, time]
            squeeze_batch = True
        
        # Convert to torch tensor
        mel_tensor = torch.from_numpy(mel).float().to(self.device)
        
        # Generate audio
        with torch.no_grad():
            audio_tensor = self.model(mel_tensor)  # [batch, 1, samples]
        
        # Convert back to numpy and remove channel dimension
        audio = audio_tensor.cpu().numpy()
        audio = audio.squeeze(1)  # [batch, samples]
        
        # Remove batch dimension if input was 2D
        if squeeze_batch:
            audio = audio[0]  # [samples]
        
        return audio


# Global vocoder instance (lazy loaded)
_vocoder_instance = None
_vocoder_checkpoint_path = None


def get_pretrained_hifigan(
    checkpoint_path: Optional[Union[str, Path]] = None,
    force_reload: bool = False
) -> HiFiGANGenerator:
    """
    Get or create a pre-trained HiFiGAN vocoder instance.
    
    This function uses a singleton pattern to avoid reloading the model
    multiple times.
    
    Args:
        checkpoint_path: Path to checkpoint. If None, uses default location.
        force_reload: Force reload even if model is already loaded.
    
    Returns:
        HiFiGANGenerator instance
    """
    global _vocoder_instance, _vocoder_checkpoint_path
    
    # Default checkpoint path
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent.parent.parent / "models" / "hifigan" / \
                         "models--speechbrain--tts-hifigan-ljspeech" / "snapshots" / \
                         "17fbdc3aae35b81e1554111fa54eab5f2b70cedb" / "generator.ckpt"
    
    checkpoint_path = Path(checkpoint_path)
    
    # Check if we need to load/reload
    if force_reload or _vocoder_instance is None or _vocoder_checkpoint_path != checkpoint_path:
        logger.info("Initializing HiFiGAN vocoder...")
        _vocoder_instance = HiFiGANGenerator(checkpoint_path)
        _vocoder_checkpoint_path = checkpoint_path
    
    return _vocoder_instance


def infer_hifigan(
    mel: np.ndarray,
    sample_rate: Optional[int] = None,
    hop_length: Optional[int] = None,
    checkpoint_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """
    Inference function compatible with synthesize.py script.
    
    This is the main entry point that should be used with --vocoder_entry.
    
    Args:
        mel: Mel-spectrogram [batch, n_mels, time] or [n_mels, time]
        sample_rate: Sample rate (unused, for compatibility)
        hop_length: Hop length (unused, for compatibility)
        checkpoint_path: Optional checkpoint path
    
    Returns:
        audio: Waveform as numpy array
    
    Example:
        In synthesize.py, use:
        --vocoder hifigan --vocoder_entry iris.hifigan_pretrained:infer_hifigan
    """
    vocoder = get_pretrained_hifigan(checkpoint_path)
    audio = vocoder(mel)
    
    # Ensure output is 1D for single batch
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    
    return audio

