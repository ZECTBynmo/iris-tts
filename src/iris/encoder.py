"""Phoneme encoder with Transformer architecture for TTS."""

import keras
from keras import layers, ops
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Optional, Tuple


class PositionalEmbedding(layers.Layer):
    """Learned positional embedding layer."""
    
    def __init__(self, max_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(
            input_dim=max_length,
            output_dim=embed_dim
        )
    
    def call(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            
        Returns:
            x + positional embeddings [batch, seq_len, embed_dim]
        """
        seq_len = ops.shape(x)[1]
        positions = ops.arange(0, seq_len, dtype='int32')
        positions = ops.expand_dims(positions, 0)  # [1, seq_len]
        position_embeddings = self.position_embedding(positions)
        return x + position_embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerBlock(layers.Layer):
    """Transformer encoder block with self-attention and feed-forward network."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        
        # Self-attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.attention_dropout = layers.Dropout(dropout)
        self.attention_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(ffn_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        self.ffn_dropout = layers.Dropout(dropout)
        self.ffn_norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, training=False, mask=None):
        """
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            training: Whether in training mode
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, attention_mask=mask, training=training)
        attn_output = self.attention_dropout(attn_output, training=training)
        x = self.attention_norm(x + attn_output)
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        x = self.ffn_norm(x + ffn_output)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout_rate,
        })
        return config


class PhonemeEncoder(keras.Model):
    """
    Transformer-based phoneme encoder for TTS.
    
    Converts phoneme sequences into contextualized representations
    that capture phoneme identity, surrounding context, and linguistic features.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_blocks: int = 4,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        max_length: int = 1000,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize phoneme encoder.
        
        Args:
            vocab_size: Size of phoneme vocabulary (including special tokens)
            embed_dim: Embedding dimension
            num_blocks: Number of transformer blocks
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network hidden dimension (default: 4 * embed_dim)
            max_length: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim or (4 * embed_dim)
        self.max_length = max_length
        self.dropout_rate = dropout
        
        # Phoneme embedding
        self.phoneme_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name='phoneme_embedding'
        )
        
        # Positional encoding
        self.positional_embedding = PositionalEmbedding(
            max_length=max_length,
            embed_dim=embed_dim
        )
        
        self.embedding_dropout = layers.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=self.ffn_dim,
                dropout=dropout,
                name=f'transformer_block_{i}'
            )
            for i in range(num_blocks)
        ]
        
        # Final layer normalization
        self.final_norm = layers.LayerNormalization(epsilon=1e-6, name='encoder_output_norm')
    
    def call(self, phoneme_ids, training=False, mask=None):
        """
        Forward pass through encoder.
        
        Args:
            phoneme_ids: Phoneme ID sequence [batch, seq_len]
            training: Whether in training mode
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Encoder output [batch, seq_len, embed_dim]
        """
        # Embed phonemes
        x = self.phoneme_embedding(phoneme_ids)  # [batch, seq_len, embed_dim]
        
        # Add positional information
        x = self.positional_embedding(x)
        x = self.embedding_dropout(x, training=training)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training, mask=mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_blocks": self.num_blocks,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "max_length": self.max_length,
            "dropout": self.dropout_rate,
        })
        return config


class DurationPredictor(keras.Model):
    """
    Duration prediction head that predicts phoneme durations from encoder outputs.
    
    Uses convolutional layers to capture local context for duration prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize duration predictor.
        
        Args:
            hidden_dim: Hidden dimension for conv layers
            num_layers: Number of conv layers
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super().__init__(**kwargs)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        # Convolutional layers
        self.conv_layers = []
        self.norm_layers = []
        self.dropout_layers = []
        
        for i in range(num_layers):
            self.conv_layers.append(
                layers.Conv1D(
                    filters=hidden_dim,
                    kernel_size=kernel_size,
                    padding='same',
                    activation='relu',
                    name=f'duration_conv_{i}'
                )
            )
            self.norm_layers.append(
                layers.LayerNormalization(epsilon=1e-6, name=f'duration_norm_{i}')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout, name=f'duration_dropout_{i}')
            )
        
        # Output projection (1 value per phoneme)
        self.output_proj = layers.Conv1D(
            filters=1,
            kernel_size=1,
            name='duration_output'
        )
    
    def call(self, encoder_output, training=False):
        """
        Predict durations from encoder output.
        
        Args:
            encoder_output: Encoder representations [batch, seq_len, embed_dim]
            training: Whether in training mode
            
        Returns:
            Log-durations [batch, seq_len, 1]
        """
        x = encoder_output
        
        # Process through conv layers
        for conv, norm, dropout in zip(self.conv_layers, self.norm_layers, self.dropout_layers):
            x = conv(x)
            x = norm(x)
            x = dropout(x, training=training)
        
        # Output projection (log-space durations)
        x = self.output_proj(x)  # [batch, seq_len, 1]
        
        # Apply softplus to keep outputs positive and smooth
        # softplus(x) = log(1 + exp(x)), always positive with smooth gradients
        # Ranges from ~0 to infinity smoothly
        x = ops.softplus(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_rate,
        })
        return config


def create_encoder(
    vocab_size: int,
    embed_dim: int = 256,
    num_blocks: int = 4,
    num_heads: int = 4,
    **kwargs
) -> PhonemeEncoder:
    """
    Create a phoneme encoder instance with default settings.
    
    Args:
        vocab_size: Size of phoneme vocabulary
        embed_dim: Embedding dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        **kwargs: Additional arguments for PhonemeEncoder
        
    Returns:
        PhonemeEncoder instance
    """
    return PhonemeEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        **kwargs
    )


def create_duration_predictor(
    hidden_dim: int = 256,
    **kwargs
) -> DurationPredictor:
    """
    Create a duration predictor instance with default settings.
    
    Args:
        hidden_dim: Hidden dimension
        **kwargs: Additional arguments for DurationPredictor
        
    Returns:
        DurationPredictor instance
    """
    return DurationPredictor(hidden_dim=hidden_dim, **kwargs)


# ============================================
# JAX JIT-compiled Utility Functions
# ============================================

@jax.jit
def length_regulate(encoder_output: jnp.ndarray, durations: jnp.ndarray) -> jnp.ndarray:
    """
    Expand encoder outputs by predicted durations (JIT-compiled for speed).
    
    This is used to convert phoneme-level representations to frame-level
    representations by repeating each phoneme encoding by its duration.
    
    Args:
        encoder_output: Encoder representations [batch, seq_len, hidden_dim]
        durations: Duration per phoneme [batch, seq_len] (integer counts)
        
    Returns:
        Expanded representations [batch, total_frames, hidden_dim]
        
    Example:
        encoder_output: [[e1, e2, e3]]  # 3 phonemes
        durations: [[2, 3, 1]]          # frames per phoneme
        output: [[e1, e1, e2, e2, e2, e3]]  # 6 frames total
    """
    batch_size, seq_len, hidden_dim = encoder_output.shape
    
    # Compute cumulative sum to get repeat indices
    # This uses JAX operations that compile efficiently
    max_len = jnp.sum(jnp.max(durations, axis=0))
    
    def expand_sequence(encoder_seq, duration_seq):
        """Expand a single sequence."""
        # Create indices for gathering
        indices = jnp.repeat(jnp.arange(seq_len), duration_seq, total_repeat_length=max_len)
        # Pad to max_len if needed
        padded = jnp.pad(indices, (0, max_len - indices.shape[0]), constant_values=0)
        # Gather encoder outputs
        return encoder_seq[padded[:max_len]]
    
    # Vectorized map over batch (JIT-friendly)
    expanded = jax.vmap(expand_sequence)(encoder_output, durations)
    
    return expanded


@partial(jax.jit, static_argnums=(1,))
def create_padding_mask(lengths: jnp.ndarray, max_len: int) -> jnp.ndarray:
    """
    Create padding mask from sequence lengths (JIT-compiled).
    
    Args:
        lengths: Sequence lengths [batch]
        max_len: Maximum sequence length (static argument - will recompile for different values)
        
    Returns:
        Boolean mask [batch, max_len] where True = valid position
    """
    positions = jnp.arange(max_len)[None, :]  # [1, max_len]
    lengths = lengths[:, None]  # [batch, 1]
    mask = positions < lengths  # [batch, max_len]
    return mask


@jax.jit
def compute_duration_loss(predicted_log_durations: jnp.ndarray, 
                          target_durations: jnp.ndarray,
                          mask: Optional[jnp.ndarray] = None,
                          delta: float = 10.0) -> jnp.ndarray:
    """
    Compute duration loss using Huber loss in linear space.
    
    Huber loss is MSE for small errors, MAE for large errors.
    This prevents gradient explosion from outliers while still
    penalizing uniform predictions more than log-space MSE.
    
    Args:
        predicted_log_durations: Predicted log-durations [batch, seq_len, 1]
        target_durations: Ground truth durations [batch, seq_len]
        mask: Optional mask for valid positions [batch, seq_len]
        delta: Threshold for switching from MSE to MAE (frames)
        
    Returns:
        Scalar loss value
    """
    # Convert predictions back to linear space
    # predicted_log_durations are log(duration+1), so invert:
    predicted_durations = jnp.exp(predicted_log_durations) - 1.0  # [batch, seq_len, 1]
    predicted_durations = predicted_durations[..., 0]  # [batch, seq_len]
    
    # Huber loss: smooth L1 loss
    # For |error| < delta: 0.5 * error^2
    # For |error| >= delta: delta * (|error| - 0.5 * delta)
    diff = predicted_durations - target_durations
    abs_diff = jnp.abs(diff)
    
    # Quadratic for small errors, linear for large
    huber = jnp.where(
        abs_diff <= delta,
        0.5 * jnp.square(diff),
        delta * (abs_diff - 0.5 * delta)
    )
    
    # Apply mask if provided
    if mask is not None:
        huber = huber * mask
        loss = jnp.sum(huber) / (jnp.sum(mask) + 1e-8)
    else:
        loss = jnp.mean(huber)
    
    return loss