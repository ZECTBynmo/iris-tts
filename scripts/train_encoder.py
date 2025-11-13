"""Training script for phoneme encoder + duration predictor."""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import argparse
import logging
from pathlib import Path
import json
import numpy as np

import keras
from keras import ops
import jax
import jax.numpy as jnp

from iris.encoder import (
    PhonemeEncoder, 
    DurationPredictor,
    compute_duration_loss,
    create_padding_mask
)
from iris.datasets import LJSpeechDurationDataset, collate_duration_batch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EncoderDurationModel(keras.Model):
    """Combined model for encoder + duration predictor."""
    
    def __init__(self, encoder, duration_predictor, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.duration_predictor = duration_predictor
    
    def call(self, phoneme_ids, training=False):
        """Forward pass."""
        encoder_out = self.encoder(phoneme_ids, training=training)
        pred_durations = self.duration_predictor(encoder_out, training=training)
        return pred_durations
    
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """Compute duration loss with masking."""
        # x is not used, y contains (durations, mask)
        target_durations, mask = y
        return compute_duration_loss(y_pred, target_durations, mask)


def train_encoder(
    ljspeech_dir: str,
    alignments_dir: str,
    output_dir: str,
    embed_dim: int = 256,
    num_blocks: int = 4,
    num_heads: int = 4,
    batch_size: int = 32,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    val_split: float = 0.05,
    checkpoint_dir: str = "checkpoints",
    log_interval: int = 100,
):
    """
    Train phoneme encoder with duration prediction.
    
    Args:
        ljspeech_dir: Path to LJSpeech dataset
        alignments_dir: Path to MFA alignments
        output_dir: Output directory for results
        embed_dim: Encoder embedding dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        val_split: Validation split fraction
        checkpoint_dir: Directory to save checkpoints
        log_interval: Steps between logging
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config = {
        'ljspeech_dir': ljspeech_dir,
        'alignments_dir': alignments_dir,
        'embed_dim': embed_dim,
        'num_blocks': num_blocks,
        'num_heads': num_heads,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'val_split': val_split,
    }
    
    config_file = output_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_file}")
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = LJSpeechDurationDataset(
        ljspeech_dir=ljspeech_dir,
        alignments_dir=alignments_dir,
        split='train',
        val_split=val_split,
        cache_dir=str(output_dir / 'cache')
    )
    
    val_dataset = LJSpeechDurationDataset(
        ljspeech_dir=ljspeech_dir,
        alignments_dir=alignments_dir,
        split='val',
        val_split=val_split,
        cache_dir=str(output_dir / 'cache')
    )
    
    vocab_size = train_dataset.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Build models
    logger.info("Building models...")
    encoder = PhonemeEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=0.1
    )
    
    duration_predictor = DurationPredictor(
        hidden_dim=embed_dim,
        dropout=0.5
    )
    
    # Build combined model
    combined_model = EncoderDurationModel(encoder, duration_predictor)
    
    # Build model with dummy input
    dummy_phoneme_ids = jnp.ones((1, 10), dtype='int32')
    _ = combined_model(dummy_phoneme_ids, training=False)
    
    logger.info(f"Encoder parameters: {encoder.count_params():,}")
    logger.info(f"Duration predictor parameters: {duration_predictor.count_params():,}")
    logger.info(f"Total parameters: {combined_model.count_params():,}")
    
    # Compile model
    combined_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_model.compute_loss,
        jit_compile=True  # Enable XLA compilation
    )
    
    logger.info("Model compiled with JIT compilation enabled")
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        epoch_losses = []
        num_batches = int(np.ceil(len(train_dataset) / batch_size))
        
        # Shuffle training data
        train_indices = np.random.permutation(len(train_dataset))
        
        for step in range(num_batches):
            # Get batch
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(train_dataset))
            batch_indices = train_indices[start_idx:end_idx]
            
            # Collect batch samples
            batch = [train_dataset[int(i)] for i in batch_indices]
            collated = collate_duration_batch(batch)
            
            # Prepare inputs
            phoneme_ids = jnp.array(collated['phoneme_ids'])
            durations = jnp.array(collated['durations'])
            lengths = jnp.array(collated['lengths'])
            
            # Create mask
            max_len = phoneme_ids.shape[1]
            mask = create_padding_mask(lengths, max_len)
            
            # Training step
            loss = combined_model.train_on_batch(
                x=phoneme_ids,
                y=(durations, mask)
            )
            
            epoch_losses.append(float(loss))
            global_step += 1
            
            # Logging
            if step % log_interval == 0:
                avg_loss = np.mean(epoch_losses[-log_interval:])
                logger.info(
                    f"Step {global_step} ({step}/{num_batches}) | "
                    f"Loss: {float(loss):.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
        
        train_loss = np.mean(epoch_losses)
        
        # Validation
        val_losses = []
        num_val_batches = int(np.ceil(len(val_dataset) / batch_size))
        
        for step in range(num_val_batches):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(val_dataset))
            
            # Collect batch samples
            batch = [val_dataset[i] for i in range(start_idx, end_idx)]
            collated = collate_duration_batch(batch)
            
            # Prepare inputs
            phoneme_ids = jnp.array(collated['phoneme_ids'])
            durations = jnp.array(collated['durations'])
            lengths = jnp.array(collated['lengths'])
            
            # Create mask
            max_len = phoneme_ids.shape[1]
            mask = create_padding_mask(lengths, max_len)
            
            # Validation step
            loss = combined_model.test_on_batch(
                x=phoneme_ids,
                y=(durations, mask)
            )
            
            val_losses.append(float(loss))
        
        val_loss = np.mean(val_losses)
        
        logger.info(
            f"Epoch {epoch + 1} Summary | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            combined_model.save_weights(checkpoint_dir / "model_best.weights.h5")
            encoder.save_weights(checkpoint_dir / "encoder_best.weights.h5")
            duration_predictor.save_weights(checkpoint_dir / "duration_best.weights.h5")
            logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            combined_model.save_weights(checkpoint_dir / f"model_epoch_{epoch+1}.weights.h5")
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    combined_model.save_weights(checkpoint_dir / "model_final.weights.h5")
    encoder.save_weights(checkpoint_dir / "encoder_final.weights.h5")
    duration_predictor.save_weights(checkpoint_dir / "duration_final.weights.h5")
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train phoneme encoder')
    parser.add_argument(
        '--ljspeech_dir',
        type=str,
        default='data/LJSpeech-1.1',
        help='Path to LJSpeech dataset'
    )
    parser.add_argument(
        '--alignments_dir',
        type=str,
        default='data/ljspeech_alignments/LJSpeech',
        help='Path to MFA alignments'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/encoder',
        help='Output directory'
    )
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.05)
    
    args = parser.parse_args()
    
    train_encoder(
        ljspeech_dir=args.ljspeech_dir,
        alignments_dir=args.alignments_dir,
        output_dir=args.output_dir,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        val_split=args.val_split,
    )


if __name__ == '__main__':
    main()

