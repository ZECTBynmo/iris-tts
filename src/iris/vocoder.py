"""HiFiGAN Vocoder implemented in Keras with JAX backend."""

import keras
from keras import layers, ops
import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ResBlock(layers.Layer):
    """Residual block with dilated convolutions."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int, ...] = (1, 3, 5), **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        
        # Create conv layers for each dilation
        self.convs1 = []
        self.convs2 = []
        for d in dilations:
            self.convs1.append(
                layers.Conv1D(channels, kernel_size, dilation_rate=d, padding='same')
            )
            self.convs2.append(
                layers.Conv1D(channels, kernel_size, padding='same')
            )
    
    def call(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = ops.leaky_relu(x, negative_slope=0.1)
            xt = c1(xt)
            xt = ops.leaky_relu(xt, negative_slope=0.1)
            xt = c2(xt)
            x = xt + x
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
        })
        return config


class HiFiGANGenerator(keras.Model):
    """
    HiFiGAN Generator implemented in Keras/JAX.
    
    Converts mel-spectrograms to audio waveforms.
    """
    
    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilations = resblock_dilations
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Pre-convolution
        self.conv_pre = layers.Conv1D(upsample_initial_channel, 7, padding='same')
        
        # Upsampling layers
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2**(i + 1))
            self.ups.append(
                layers.Conv1DTranspose(out_ch, k, strides=u, padding='same')
            )
        
        # Residual blocks
        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # Post-convolution
        self.conv_post = layers.Conv1D(1, 7, padding='same', activation='tanh')
    
    def call(self, x, training=False):
        """
        Args:
            x: Mel-spectrogram [batch, time, mel_channels]
            
        Returns:
            audio: Waveform [batch, time * prod(upsample_rates), 1]
        """
        x = self.conv_pre(x)
        
        for i, up in enumerate(self.ups):
            x = ops.leaky_relu(x, negative_slope=0.1)
            x = up(x)
            
            # Multi-receptive field fusion
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels
        
        x = ops.leaky_relu(x, negative_slope=0.1)
        x = self.conv_post(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "in_channels": self.in_channels,
            "upsample_rates": self.upsample_rates,
            "upsample_kernel_sizes": self.upsample_kernel_sizes,
            "upsample_initial_channel": self.upsample_initial_channel,
            "resblock_kernel_sizes": self.resblock_kernel_sizes,
            "resblock_dilations": self.resblock_dilations,
        })
        return config


class HiFiGANVocoder:
    """High-level interface for HiFiGAN vocoder in Keras/JAX."""
    
    def __init__(self, weights_path: Optional[str] = None):
        """
        Initialize HiFiGAN vocoder.
        
        Args:
            weights_path: Path to model weights (.keras or .h5 file)
        """
        self.model = HiFiGANGenerator()
        
        # Build model with dummy input
        dummy_input = ops.ones((1, 100, 80))  # [batch, time, mel_channels]
        _ = self.model(dummy_input, training=False)
        
        if weights_path and Path(weights_path).exists():
            self.load_weights(weights_path)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.info("Initialized HiFiGAN with random weights (needs training)")
    
    def load_weights(self, weights_path: str):
        """Load model weights."""
        self.model.load_weights(weights_path)
        logger.info(f"Loaded weights from {weights_path}")
    
    def save_weights(self, weights_path: str):
        """Save model weights."""
        self.model.save_weights(weights_path)
        logger.info(f"Saved weights to {weights_path}")
    
    def infer(self, mel: np.ndarray) -> np.ndarray:
        """
        Convert mel-spectrogram to audio waveform.
        
        Args:
            mel: Mel-spectrogram [mel_channels, time] or [batch, mel_channels, time]
            
        Returns:
            audio: Waveform [samples] or [batch, samples]
        """
        # Handle different input shapes
        squeeze_batch = False
        squeeze_channel = False
        
        if mel.ndim == 2:
            # [mel_channels, time] -> [batch, time, mel_channels]
            mel = mel.T[np.newaxis, ...]
            squeeze_batch = True
        elif mel.ndim == 3:
            # [batch, mel_channels, time] -> [batch, time, mel_channels]
            mel = np.transpose(mel, (0, 2, 1))
        
        # Generate audio
        audio = self.model(mel, training=False)
        audio = np.array(audio)
        
        # Remove channel dimension and reshape
        audio = audio[..., 0]  # [batch, time]
        
        if squeeze_batch:
            audio = audio[0]  # [time]
        
        return audio
    
    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """Convenience method for inference."""
        return self.infer(mel)


def create_vocoder(weights_path: Optional[str] = None) -> HiFiGANVocoder:
    """
    Create a HiFiGAN vocoder instance.
    
    Args:
        weights_path: Path to pre-trained weights (optional)
        
    Returns:
        HiFiGANVocoder instance
    """
    return HiFiGANVocoder(weights_path=weights_path)

