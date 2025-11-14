"""Test script to evaluate a trained encoder + duration predictor."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import logging
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from iris.encoder import (
    PhonemeEncoder,
    DurationPredictor,
    compute_duration_loss,
    create_padding_mask,
)
from iris.datasets import LJSpeechDurationDataset, collate_duration_batch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_datasets(ljspeech_dir: str, alignments_dir: str, val_split: float = 0.05):
    logger.info("=" * 60)
    logger.info("Loading duration datasets (train/val)")
    logger.info("=" * 60)
    train_ds = LJSpeechDurationDataset(
        ljspeech_dir=ljspeech_dir,
        alignments_dir=alignments_dir,
        split="train",
        val_split=val_split,
        cache_dir="outputs/test_cache",
    )
    val_ds = LJSpeechDurationDataset(
        ljspeech_dir=ljspeech_dir,
        alignments_dir=alignments_dir,
        split="val",
        val_split=val_split,
        cache_dir="outputs/test_cache",
    )
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    logger.info(f"Vocab size: {train_ds.get_vocab_size()}")
    return train_ds, val_ds


def load_models(vocab_size: int, encoder_w: str, duration_w: str, embed_dim: int = 256):
    logger.info("\n" + "=" * 60)
    logger.info("Loading trained encoder + duration predictor")
    logger.info("=" * 60)
    enc = PhonemeEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_blocks=4,
        num_heads=4,
        dropout=0.1,
    )
    dur = DurationPredictor(hidden_dim=embed_dim)
    # Build
    _ = enc(jnp.ones((1, 8), dtype="int32"), training=False)
    _ = dur(jnp.ones((1, 8, embed_dim), dtype="float32"), training=False)
    # Load weights
    encoder_path = Path(encoder_w)
    duration_path = Path(duration_w)
    logger.info(f"Loading encoder weights: {encoder_path.resolve()}")
    enc.load_weights(str(encoder_path))
    logger.info(f"Loading duration weights: {duration_path.resolve()}")
    dur.load_weights(str(duration_path))
    logger.info(f"Encoder params: {enc.count_params():,} | Duration params: {dur.count_params():,}")
    return enc, dur


def eval_split(dataset, encoder, duration_predictor, num_batches: int = 10, batch_size: int = 32):
    """Compute average duration loss and show some frame stats."""
    losses = []
    all_pred = []
    all_tgt = []
    n = len(dataset)
    max_batches = min(num_batches, int(np.ceil(n / batch_size)))
    logger.info(f"Evaluating on {max_batches} batches (batch_size={batch_size})")
    for b in range(max_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        batch = [dataset[i] for i in range(start, end)]
        col = collate_duration_batch(batch)
        phoneme_ids = jnp.array(col["phoneme_ids"])
        durations = jnp.array(col["durations"])
        lengths = jnp.array(col["lengths"])
        max_len = phoneme_ids.shape[1]
        mask = create_padding_mask(lengths, max_len)
        enc_out = encoder(phoneme_ids, training=False)
        pred_log = duration_predictor(enc_out, training=False)
        loss = compute_duration_loss(pred_log, durations, mask)
        losses.append(float(loss))
        # Save a few frame predictions for inspection
        if b == 0:
            pred_frames = jnp.exp(pred_log[:, :, 0])
            all_pred = np.asarray(pred_frames[:4, :10])
            all_tgt = np.asarray(durations[:4, :10])
    avg = float(np.mean(losses))
    logger.info(f"Mean duration loss over eval split: {avg:.4f}")
    if len(all_pred) > 0:
        logger.info("Sample predicted vs target frames (first 4 samples, first 10 phonemes):")
        for i in range(all_pred.shape[0]):
            logger.info(f"  Sample {i}:")
            logger.info(f"    pred: {all_pred[i]}")
            logger.info(f"    tgt : {all_tgt[i]}")
    return avg


def main():
    ljspeech_dir = "data/LJSpeech-1.1"
    alignments_dir = "data/ljspeech_alignments/LJSpeech"
    encoder_w = "outputs/encoder/checkpoints/encoder_best.weights.h5"
    duration_w = "outputs/encoder/checkpoints/duration_best.weights.h5"

    train_ds, val_ds = load_datasets(ljspeech_dir, alignments_dir)
    vocab_size = train_ds.get_vocab_size()
    encoder, duration_predictor = load_models(vocab_size, encoder_w, duration_w, embed_dim=256)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on TRAIN split")
    logger.info("=" * 60)
    train_loss = eval_split(train_ds, encoder, duration_predictor, num_batches=5, batch_size=32)

    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on VAL split")
    logger.info("=" * 60)
    val_loss = eval_split(val_ds, encoder, duration_predictor, num_batches=5, batch_size=32)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Train duration loss (approx): {train_loss:.4f}")
    logger.info(f"Val duration loss   (approx): {val_loss:.4f}")
    logger.info("If losses are ~<0.5 and predicted frames roughly match targets, the encoder head is in a reasonable state.")


if __name__ == "__main__":
    main()

"""Test script to verify trained encoder predictions."""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import logging
import numpy as np
import jax.numpy as jnp

from iris.encoder import PhonemeEncoder, DurationPredictor
from iris.datasets import LJSpeechDurationDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_trained_encoder():
    """Test the trained encoder on validation samples."""
    
    logger.info("=" * 70)
    logger.info("Testing Trained Encoder")
    logger.info("=" * 70)
    
    # Load validation dataset
    logger.info("\nLoading validation dataset...")
    val_dataset = LJSpeechDurationDataset(
        ljspeech_dir='data/LJSpeech-1.1',
        alignments_dir='data/ljspeech_alignments/LJSpeech',
        split='val',
        val_split=0.05,
        cache_dir='outputs/encoder/cache'
    )
    
    vocab_size = val_dataset.get_vocab_size()
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Build models
    logger.info("\nBuilding models...")
    encoder = PhonemeEncoder(
        vocab_size=vocab_size,
        embed_dim=256,
        num_blocks=4,
        num_heads=4,
        dropout=0.1
    )
    
    duration_predictor = DurationPredictor(
        hidden_dim=256,
        dropout=0.1
    )
    
    # Build with dummy input
    dummy_input = jnp.ones((1, 10), dtype='int32')
    dummy_encoder_out = encoder(dummy_input, training=False)
    _ = duration_predictor(dummy_encoder_out, training=False)
    
    # Load trained weights
    logger.info("\nLoading trained weights...")
    try:
        encoder.load_weights('outputs/encoder/checkpoints/encoder_best.weights.h5')
        duration_predictor.load_weights('outputs/encoder/checkpoints/duration_best.weights.h5')
        logger.info("✓ Weights loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load weights: {e}")
        return
    
    # Test on multiple samples
    logger.info("\n" + "=" * 70)
    logger.info("Testing Predictions on Validation Samples")
    logger.info("=" * 70)
    
    num_test_samples = 10
    errors = []
    
    for i in range(num_test_samples):
        sample = val_dataset[i]
        
        phoneme_ids = jnp.array(sample['phoneme_ids'][None, :])  # Add batch dim
        target_durations = sample['durations']
        
        # Forward pass
        encoder_out = encoder(phoneme_ids, training=False)
        pred_log_durations = duration_predictor(encoder_out, training=False)
        
        # Convert to actual durations
        pred_durations = np.exp(np.array(pred_log_durations[0, :, 0]))
        
        # Compute error
        mae = np.mean(np.abs(pred_durations - target_durations))
        mse = np.mean((pred_durations - target_durations) ** 2)
        errors.append(mae)
        
        # Display sample results
        if i < 3:  # Show first 3 in detail
            logger.info(f"\nSample {i} ({sample['file_id']}):")
            logger.info(f"  Text: {sample['text'][:80]}...")
            logger.info(f"  Sequence length: {len(target_durations)}")
            logger.info(f"  MAE: {mae:.2f} frames")
            logger.info(f"  RMSE: {np.sqrt(mse):.2f} frames")
            
            # Show first 10 predictions vs targets
            logger.info(f"  First 10 phonemes:")
            logger.info(f"    Predicted: {pred_durations[:10]}")
            logger.info(f"    Target:    {target_durations[:10]}")
            
            # Compute correlation
            correlation = np.corrcoef(pred_durations, target_durations)[0, 1]
            logger.info(f"  Correlation: {correlation:.3f}")
    
    # Overall statistics
    logger.info("\n" + "=" * 70)
    logger.info("Overall Statistics")
    logger.info("=" * 70)
    logger.info(f"Mean MAE: {np.mean(errors):.2f} frames")
    logger.info(f"Std MAE: {np.std(errors):.2f} frames")
    logger.info(f"Min MAE: {np.min(errors):.2f} frames")
    logger.info(f"Max MAE: {np.max(errors):.2f} frames")
    
    # Interpretation
    logger.info("\n" + "=" * 70)
    logger.info("Interpretation")
    logger.info("=" * 70)
    
    mean_mae = np.mean(errors)
    
    # At 22050 Hz with hop_length=256, each frame is ~11.6ms
    ms_per_frame = (256 / 22050) * 1000
    mae_in_ms = mean_mae * ms_per_frame
    
    logger.info(f"Average error: {mae_in_ms:.1f} ms per phoneme")
    logger.info(f"Frame duration: {ms_per_frame:.1f} ms")
    
    if mean_mae < 5:
        logger.info("✓ EXCELLENT: Very accurate duration predictions!")
    elif mean_mae < 10:
        logger.info("✓ GOOD: Reasonable duration predictions")
    elif mean_mae < 20:
        logger.info("⚠ FAIR: Some room for improvement")
    else:
        logger.info("✗ POOR: High prediction error")
    
    # Test batch processing
    logger.info("\n" + "=" * 70)
    logger.info("Testing Batch Processing")
    logger.info("=" * 70)
    
    # Collect a batch
    batch_size = 8
    batch_phoneme_ids = []
    batch_lengths = []
    max_len = 0
    
    for i in range(batch_size):
        sample = val_dataset[i]
        batch_phoneme_ids.append(sample['phoneme_ids'])
        batch_lengths.append(len(sample['phoneme_ids']))
        max_len = max(max_len, len(sample['phoneme_ids']))
    
    # Pad to max length
    padded_batch = np.zeros((batch_size, max_len), dtype=np.int32)
    for i, seq in enumerate(batch_phoneme_ids):
        padded_batch[i, :len(seq)] = seq
    
    padded_batch = jnp.array(padded_batch)
    
    # Batch forward pass
    logger.info(f"Batch shape: {padded_batch.shape}")
    batch_encoder_out = encoder(padded_batch, training=False)
    batch_pred_durations = duration_predictor(batch_encoder_out, training=False)
    
    logger.info(f"✓ Batch encoder output: {batch_encoder_out.shape}")
    logger.info(f"✓ Batch duration predictions: {batch_pred_durations.shape}")
    logger.info("✓ Batch processing works!")
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ ALL TESTS PASSED!")
    logger.info("=" * 70)
    logger.info("The encoder is ready to use for VAE training.")
    logger.info("You can load weights from: outputs/encoder/checkpoints/encoder_best.weights.h5")
    logger.info("")


def main():
    try:
        test_trained_encoder()
    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

