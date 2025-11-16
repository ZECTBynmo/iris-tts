"""Debug VAE loss computation to understand why values are so large."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax.numpy as jnp
import keras
from keras import ops
from pathlib import Path
import logging

from iris.vae import TextConditionedVAE
from iris.encoder import PhonemeEncoder
from iris.datasets import LJSpeechVAEDataset, collate_vae_batch
from scripts.train_vae import build_frame_level_condition

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("VAE Loss Debugging - Single Batch Analysis")
    logger.info("=" * 80)
    
    # Config matching current training
    n_mels = 80
    embed_dim = 256
    down_stages = 2
    batch_size = 8
    
    # Load dataset
    logger.info("\n[1/5] Loading dataset...")
    dataset = LJSpeechVAEDataset(
        ljspeech_dir='data/LJSpeech-1.1',
        alignments_dir='data/ljspeech_alignments/LJSpeech',
        split='train',
        val_split=0.05,
        cache_dir='outputs/vae/cache'
    )
    
    # Get ONE small batch
    batch = [dataset[i] for i in range(batch_size)]
    col = collate_vae_batch(batch)
    
    logger.info(f"✓ Loaded batch of {batch_size} samples")
    logger.info(f"  Mel specs shape: {col['mel_specs'].shape}")
    logger.info(f"  Mel lengths: {col['mel_lengths']}")
    
    # Load encoder
    logger.info("\n[2/5] Loading encoder...")
    vocab_size = dataset.get_vocab_size()
    encoder = PhonemeEncoder(vocab_size=vocab_size, embed_dim=embed_dim, num_blocks=4, num_heads=4, dropout=0.1)
    _ = encoder(jnp.ones((1, 8), dtype="int32"), training=False)
    
    encoder_weights = "outputs/encoder/checkpoints/encoder_best.weights.h5"
    if Path(encoder_weights).exists():
        encoder.load_weights(encoder_weights)
        logger.info(f"✓ Loaded encoder from {encoder_weights}")
    
    # Load VAE
    logger.info("\n[3/5] Loading VAE...")
    vae = TextConditionedVAE(
        n_mels=n_mels,
        cond_dim=embed_dim,
        model_channels=192,
        latent_dim=16,
        num_wavenet_blocks=8,
        decoder_blocks=4,
        down_stages=down_stages,
        flow_layers=4,
        flow_hidden=64,
    )
    _ = vae(
        mels_bt_f=jnp.ones((1, n_mels, 16), dtype="float32"),
        frame_text_cond=jnp.ones((1, 16, embed_dim), dtype="float32"),
        training=False,
    )
    
    vae_weights = "outputs/vae/checkpoints_vae/vae_core_best.weights.h5"
    if Path(vae_weights).exists():
        vae.load_weights(vae_weights)
        logger.info(f"✓ Loaded VAE from {vae_weights}")
    else:
        logger.info("⚠ No weights found, using random initialization")
    
    # Prepare inputs
    logger.info("\n[4/5] Preparing inputs...")
    phoneme_ids = jnp.array(col["phoneme_ids"])
    mels_bt_f = jnp.array(col["mel_specs"])
    durations = np.array(col["durations"])
    mel_len = np.array(col["mel_lengths"])
    
    logger.info(f"  Input mel stats:")
    logger.info(f"    Range: [{float(ops.min(mels_bt_f)):.3f}, {float(ops.max(mels_bt_f)):.3f}]")
    logger.info(f"    Mean: {float(ops.mean(mels_bt_f)):.3f}")
    logger.info(f"    Std: {float(ops.std(mels_bt_f)):.3f}")
    
    # Encoder
    enc_out = encoder(phoneme_ids, training=False)
    enc_out_np = np.array(enc_out)
    
    # Frame conditioning
    frame_cond_np, mask_np = build_frame_level_condition(enc_out_np, durations, mel_len)
    
    # Pad to factor
    factor = 2 ** down_stages
    t_max = frame_cond_np.shape[1]
    target_len = int(np.ceil(t_max / factor) * factor)
    
    mels_bt_f_np = np.array(mels_bt_f)
    if mels_bt_f_np.shape[2] < target_len:
        pad = np.zeros((mels_bt_f_np.shape[0], mels_bt_f_np.shape[1], target_len - mels_bt_f_np.shape[2]), dtype=mels_bt_f_np.dtype)
        mels_bt_f_np = np.concatenate([mels_bt_f_np, pad], axis=2)
    elif mels_bt_f_np.shape[2] > target_len:
        mels_bt_f_np = mels_bt_f_np[:, :, :target_len]
    
    if frame_cond_np.shape[1] < target_len:
        pad = np.zeros((frame_cond_np.shape[0], target_len - frame_cond_np.shape[1], frame_cond_np.shape[2]), dtype=frame_cond_np.dtype)
        frame_cond_np = np.concatenate([frame_cond_np, pad], axis=1)
    if mask_np.shape[1] < target_len:
        padm = np.zeros((mask_np.shape[0], target_len - mask_np.shape[1]), dtype=mask_np.dtype)
        mask_np = np.concatenate([mask_np, padm], axis=1)
    
    frame_cond = jnp.array(frame_cond_np)
    mask_bt = jnp.array(mask_np)
    mels_bt_f = jnp.array(mels_bt_f_np)
    
    logger.info(f"  After padding to {target_len}:")
    logger.info(f"    Mels: {mels_bt_f.shape}")
    logger.info(f"    Conditioning: {frame_cond.shape}")
    logger.info(f"    Mask: {mask_bt.shape}")
    logger.info(f"    Valid frames: {[int(mask_bt[i].sum()) for i in range(batch_size)]}")
    
    # Forward pass
    logger.info("\n[5/5] Computing loss components...")
    recon, (mean, logvar), _ = vae(mels_bt_f, frame_cond, training=True)
    
    logger.info(f"  VAE output:")
    logger.info(f"    Recon shape: {recon.shape}")
    logger.info(f"    Recon range: [{float(ops.min(recon)):.3f}, {float(ops.max(recon)):.3f}]")
    logger.info(f"    Recon mean: {float(ops.mean(recon)):.3f}")
    logger.info(f"    Recon std: {float(ops.std(recon)):.3f}")
    logger.info(f"    Contains NaN: {bool(jnp.isnan(recon).any())}")
    logger.info(f"    Contains Inf: {bool(jnp.isinf(recon).any())}")
    
    logger.info(f"  Latent stats:")
    logger.info(f"    Mean shape: {mean.shape}")
    logger.info(f"    Mean range: [{float(ops.min(mean)):.3f}, {float(ops.max(mean)):.3f}]")
    logger.info(f"    Logvar range: [{float(ops.min(logvar)):.3f}, {float(ops.max(logvar)):.3f}]")
    
    # Reconstruction loss
    logger.info("\n" + "=" * 80)
    logger.info("RECONSTRUCTION LOSS BREAKDOWN:")
    logger.info("=" * 80)
    
    diff = ops.abs(mels_bt_f - recon)
    logger.info(f"  Absolute difference:")
    logger.info(f"    Shape: {diff.shape}")
    logger.info(f"    Range: [{float(ops.min(diff)):.3f}, {float(ops.max(diff)):.3f}]")
    logger.info(f"    Mean: {float(ops.mean(diff)):.3f}")
    logger.info(f"    Sum: {float(ops.sum(diff)):.2e}")
    
    # Apply mask
    m = ops.expand_dims(mask_bt, axis=1)  # [B, 1, T]
    diff_masked = diff * m
    
    logger.info(f"  After masking:")
    logger.info(f"    Masked diff sum: {float(ops.sum(diff_masked)):.2e}")
    logger.info(f"    Mask sum: {float(ops.sum(m)):.2e}")
    logger.info(f"    N_mels: {float(ops.cast(ops.shape(diff)[1], diff.dtype))}")
    
    denominator = ops.sum(m) * ops.cast(ops.shape(diff)[1], diff.dtype) + 1e-6
    loss_recon = ops.sum(diff_masked) / denominator
    
    logger.info(f"  Denominator: {float(denominator):.2e}")
    logger.info(f"  ✓ Reconstruction L1 loss: {float(loss_recon):.6f}")
    
    # KL loss
    logger.info("\n" + "=" * 80)
    logger.info("KL DIVERGENCE LOSS BREAKDOWN:")
    logger.info("=" * 80)
    
    kl = -0.5 * (1 + logvar - ops.square(mean) - ops.exp(logvar))
    logger.info(f"  KL per element:")
    logger.info(f"    Shape: {kl.shape}")
    logger.info(f"    Mean: {float(ops.mean(kl)):.6f}")
    logger.info(f"    Sum: {float(ops.sum(kl)):.2e}")
    
    # Downsample mask
    mask_downsampled = mask_bt[:, ::factor]
    m_kl = ops.expand_dims(mask_downsampled, axis=-1)
    kl_masked = kl * m_kl
    loss_kl = ops.sum(kl_masked) / (ops.sum(m_kl) + 1e-8)
    
    logger.info(f"  After masking:")
    logger.info(f"    Masked KL sum: {float(ops.sum(kl_masked)):.2e}")
    logger.info(f"    Mask sum: {float(ops.sum(m_kl)):.2e}")
    logger.info(f"  ✓ KL divergence loss: {float(loss_kl):.6f}")
    
    # Total loss
    kl_weight = 0.028  # Epoch 3
    total_loss = loss_recon + kl_weight * loss_kl
    
    logger.info("\n" + "=" * 80)
    logger.info("TOTAL LOSS:")
    logger.info("=" * 80)
    logger.info(f"  Reconstruction loss: {float(loss_recon):.6f}")
    logger.info(f"  KL loss: {float(loss_kl):.6f}")
    logger.info(f"  KL weight: {kl_weight:.4f}")
    logger.info(f"  Weighted KL: {float(kl_weight * loss_kl):.6f}")
    logger.info(f"  ✓ TOTAL: {float(total_loss):.6f}")
    logger.info("\n" + "=" * 80)
    logger.info(f"Expected total loss: ~{float(total_loss):.2f}")
    logger.info(f"Your training shows: ~5.2e12")
    logger.info("=" * 80)
    logger.info("\nIf these don't match, there's a bug in the training loop loss aggregation.")


if __name__ == "__main__":
    main()

