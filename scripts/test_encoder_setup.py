"""Test script to verify encoder setup before training."""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import logging
import jax.numpy as jnp

from iris.encoder import PhonemeEncoder, DurationPredictor, compute_duration_loss, create_padding_mask
from iris.datasets import LJSpeechDurationDataset, collate_duration_batch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset():
    """Test that dataset loads correctly."""
    logger.info("=" * 60)
    logger.info("Testing Dataset Loading")
    logger.info("=" * 60)
    
    try:
        # Create dataset
        dataset = LJSpeechDurationDataset(
            ljspeech_dir='data/LJSpeech-1.1',
            alignments_dir='data/ljspeech_alignments/LJSpeech',
            split='train',
            val_split=0.05,
            cache_dir='outputs/test_cache'
        )
        
        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
        logger.info(f"✓ Vocabulary size: {dataset.get_vocab_size()}")
        
        # Test getting a single sample
        sample = dataset[0]
        logger.info(f"✓ Sample 0:")
        logger.info(f"  - File ID: {sample['file_id']}")
        logger.info(f"  - Text: {sample['text'][:80]}...")
        logger.info(f"  - Phoneme IDs shape: {sample['phoneme_ids'].shape}")
        logger.info(f"  - Durations shape: {sample['durations'].shape}")
        logger.info(f"  - Length: {sample['length']}")
        
        # Test batching
        batch = [dataset[i] for i in range(4)]
        collated = collate_duration_batch(batch)
        logger.info(f"✓ Batch collation:")
        logger.info(f"  - Phoneme IDs: {collated['phoneme_ids'].shape}")
        logger.info(f"  - Durations: {collated['durations'].shape}")
        logger.info(f"  - Lengths: {collated['lengths']}")
        
        return dataset
        
    except Exception as e:
        logger.error(f"✗ Dataset loading failed: {e}")
        raise


def test_models(dataset):
    """Test that models build and forward pass works."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Model Architecture")
    logger.info("=" * 60)
    
    try:
        vocab_size = dataset.get_vocab_size()
        
        # Build models
        encoder = PhonemeEncoder(
            vocab_size=vocab_size,
            embed_dim=256,
            num_blocks=4,
            num_heads=4
        )
        
        duration_predictor = DurationPredictor(
            hidden_dim=256
        )
        
        logger.info(f"✓ Models created")
        
        # Test forward pass with single sample
        sample = dataset[0]
        phoneme_ids = jnp.array(sample['phoneme_ids'][None, :])  # Add batch dim
        
        encoder_out = encoder(phoneme_ids, training=False)
        logger.info(f"✓ Encoder forward pass: {phoneme_ids.shape} -> {encoder_out.shape}")
        
        pred_durations = duration_predictor(encoder_out, training=False)
        logger.info(f"✓ Duration predictor forward pass: {encoder_out.shape} -> {pred_durations.shape}")
        
        # Test loss computation
        target_durations = jnp.array(sample['durations'][None, :])  # Add batch dim
        mask = jnp.ones((1, phoneme_ids.shape[1]), dtype='bool')
        
        loss = compute_duration_loss(pred_durations, target_durations, mask)
        logger.info(f"✓ Loss computation: {float(loss):.4f}")
        
        # Model summary
        logger.info(f"✓ Encoder parameters: {encoder.count_params():,}")
        logger.info(f"✓ Duration predictor parameters: {duration_predictor.count_params():,}")
        
        return encoder, duration_predictor
        
    except Exception as e:
        logger.error(f"✗ Model testing failed: {e}")
        raise


def test_batched_training_step(dataset, encoder, duration_predictor):
    """Test a batched training step."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Batched Training Step")
    logger.info("=" * 60)
    
    try:
        # Create a batch
        batch = [dataset[i] for i in range(4)]
        collated = collate_duration_batch(batch)
        
        phoneme_ids = jnp.array(collated['phoneme_ids'])
        durations = jnp.array(collated['durations'])
        lengths = jnp.array(collated['lengths'])
        
        # Create mask
        max_len = phoneme_ids.shape[1]
        mask = create_padding_mask(lengths, max_len)
        
        logger.info(f"✓ Batch prepared:")
        logger.info(f"  - Phoneme IDs: {phoneme_ids.shape}")
        logger.info(f"  - Durations: {durations.shape}")
        logger.info(f"  - Mask: {mask.shape}")
        
        # Forward pass (without mask for now - padding is handled by loss)
        encoder_out = encoder(phoneme_ids, training=False)
        pred_durations = duration_predictor(encoder_out, training=False)
        loss = compute_duration_loss(pred_durations, durations, mask)
        
        logger.info(f"✓ Batched forward pass successful")
        logger.info(f"  - Encoder output: {encoder_out.shape}")
        logger.info(f"  - Predicted durations: {pred_durations.shape}")
        logger.info(f"  - Loss: {float(loss):.4f}")
        
        # Check predictions are reasonable
        pred_frames = jnp.exp(pred_durations[:, :, 0])
        target_frames = durations
        
        logger.info(f"  - Predicted frames (first sample, first 5): {pred_frames[0, :5]}")
        logger.info(f"  - Target frames (first sample, first 5): {target_frames[0, :5]}")
        
    except Exception as e:
        logger.error(f"✗ Batched training step failed: {e}")
        raise


def main():
    logger.info("Testing Encoder Setup")
    logger.info("This will verify:")
    logger.info("  1. Dataset loads MFA alignments correctly")
    logger.info("  2. Models build without errors")
    logger.info("  3. Forward pass works with batched data")
    logger.info("")
    
    try:
        # Test dataset
        dataset = test_dataset()
        
        # Test models
        encoder, duration_predictor = test_models(dataset)
        
        # Test batched training step
        test_batched_training_step(dataset, encoder, duration_predictor)
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("You're ready to start training with:")
        logger.info("  uv run python scripts/train_encoder.py")
        logger.info("")
        
    except Exception as e:
        logger.info("\n" + "=" * 60)
        logger.info("✗ TESTS FAILED")
        logger.info("=" * 60)
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()

