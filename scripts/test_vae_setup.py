"""Test script to validate VAE model structure and forward paths."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import logging
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import keras

from iris.datasets import LJSpeechVAEDataset, collate_vae_batch
from iris.encoder import PhonemeEncoder
from iris.vae import TextConditionedVAE


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_frame_level_condition(encoder_outputs, durations, mel_lengths):
	"""Map phoneme-level encoder outputs to frame-level by durations.
	
	Args:
		encoder_outputs: [B, P, E] numpy array
		durations: [B, P] numpy array of ints
		mel_lengths: [B] numpy array, target number of frames per sample
	Returns:
		frame_cond: [B, T_max, E] numpy array (padded to batch max T with zeros)
		mask: [B, T_max] boolean numpy array of valid frames
	"""
	batch_size, _, embed_dim = encoder_outputs.shape
	t_targets = mel_lengths.astype(np.int32)
	t_max = int(t_targets.max())
	frame_cond = np.zeros((batch_size, t_max, embed_dim), dtype=np.float32)
	mask = np.zeros((batch_size, t_max), dtype=bool)
	
	for i in range(batch_size):
		p_len = durations[i].shape[0]
		ph_durs = durations[i].astype(np.int32)
		cum = np.cumsum(ph_durs)
		t_len = int(t_targets[i])
		indices = np.searchsorted(cum, np.arange(t_len), side="right")
		indices = np.clip(indices, 0, p_len - 1)
		enc_i = encoder_outputs[i]
		frame_cond[i, :t_len] = enc_i[indices]
		mask[i, :t_len] = True
	
	return frame_cond, mask


def test_dataset(ljspeech_dir: str, alignments_dir: str, val_split: float = 0.05):
	logger.info("=" * 60)
	logger.info("Testing VAE Dataset Loading")
	logger.info("=" * 60)
	try:
		ds = LJSpeechVAEDataset(
			ljspeech_dir=ljspeech_dir,
			alignments_dir=alignments_dir,
			split="train",
			val_split=val_split,
		)
		logger.info(f"✓ VAE dataset loaded: {len(ds)} samples")
		logger.info(f"✓ Vocabulary size: {ds.get_vocab_size()}")
		
		# Example sample
		sample = ds[0]
		logger.info(f"  - File ID: {sample['file_id']}")
		logger.info(f"  - Text: {sample['text'][:80]}...")
		logger.info(f"  - Phoneme IDs: {sample['phoneme_ids'].shape}")
		logger.info(f"  - Mel: {sample['mel_spec'].shape} (n_mels, time)")
		logger.info(f"  - Durations: {sample['durations'].shape}")
		return ds
	except Exception as e:
		logger.error(f"✗ Dataset test failed: {e}")
		raise


def test_models(ds, encoder_weights: str = "", n_mels: int = 80, embed_dim: int = 256):
	logger.info("\n" + "=" * 60)
	logger.info("Building Encoder and VAE")
	logger.info("=" * 60)
	try:
		vocab_size = ds.get_vocab_size()
		# Text encoder for conditioning
		text_encoder = PhonemeEncoder(
			vocab_size=vocab_size,
			embed_dim=embed_dim,
			num_blocks=4,
			num_heads=4,
			dropout=0.1,
		)
		_ = text_encoder(jnp.ones((1, 8), dtype="int32"), training=False)
		if encoder_weights and Path(encoder_weights).exists():
			logger.info(f"Loading encoder weights from {encoder_weights}")
			text_encoder.load_weights(encoder_weights)
		else:
			if encoder_weights:
				logger.warning(f"Encoder weights not found: {encoder_weights}, proceeding without.")
			else:
				logger.info("No encoder weights provided; using randomly initialized encoder.")
		
		# VAE
		vae = TextConditionedVAE(
			n_mels=n_mels,
			cond_dim=embed_dim,
			model_channels=embed_dim,
			num_wavenet_blocks=4,
			down_stages=2,
			flow_layers=2,
			flow_hidden=embed_dim,
		)
		_ = vae(
			mels_bt_f=jnp.ones((1, n_mels, 16), dtype="float32"),
			frame_text_cond=jnp.ones((1, 16, embed_dim), dtype="float32"),
			training=False,
		)
		logger.info(f"✓ VAE built | params: {vae.count_params():,}")
		return text_encoder, vae
	except Exception as e:
		logger.error(f"✗ Model build failed: {e}")
		raise


def test_forward_batch(ds, text_encoder, vae, batch_size: int = 2):
	logger.info("\n" + "=" * 60)
	logger.info("Testing Forward Pass")
	logger.info("=" * 60)
	try:
		# Prepare batch
		batch = [ds[i] for i in range(min(batch_size, len(ds)))]
		col = collate_vae_batch(batch)
		
		phoneme_ids = jnp.array(col["phoneme_ids"])
		mels_bt_f = jnp.array(col["mel_specs"])
		durations = np.array(col["durations"])
		mel_len = np.array(col["mel_lengths"])
		
		# Encoder -> phoneme-level conditioning
		enc_out = text_encoder(phoneme_ids, training=False)  # [B, P, E]
		enc_out_np = np.array(enc_out)
		
		# Frame-level conditioning and mask
		frame_cond_np, mask_np = build_frame_level_condition(enc_out_np, durations, mel_len)
		t_max = frame_cond_np.shape[1]
		
		# Pad/trim mels to T_max
		def pad_or_trim_mels(x_bt_f, target_t):
			b, f, t = x_bt_f.shape
			if t == target_t:
				return x_bt_f
			if t > target_t:
				return x_bt_f[:, :, :target_t]
			pad = np.zeros((b, f, target_t - t), dtype=x_bt_f.dtype)
			return np.concatenate([x_bt_f, pad], axis=2)
		
		mels_bt_f_np = np.array(mels_bt_f)
		mels_bt_f_np = pad_or_trim_mels(mels_bt_f_np, t_max)
		
		# To jnp
		frame_cond = jnp.array(frame_cond_np)
		mask_bt = jnp.array(mask_np)
		mels_bt_f = jnp.array(mels_bt_f_np)
		
		# Forward
		recon, (mean, logvar), residual = vae(mels_bt_f, frame_cond, training=False)
		logger.info(f"✓ Forward pass:")
		logger.info(f"  - Input mels: {mels_bt_f.shape} [B, n_mels, T]")
		logger.info(f"  - Frame cond: {frame_cond.shape} [B, T, E]")
		logger.info(f"  - Recon mels: {recon.shape} [B, n_mels, T]")
		logger.info(f"  - Posterior mean/logvar: {mean.shape}, {logvar.shape} [B, T', C]")
		if residual is not None:
			logger.info(f"  - Residual embedding: {residual.shape} [B, T, E]")
		
		# Simple losses
		l1 = vae.compute_recon_l1(mels_bt_f, recon, mask=mask_bt)
		kl = vae.compute_kl(mean, logvar)
		logger.info(f"  - Recon L1: {float(l1):.4f} | KL: {float(kl):.4f}")
		
		# Generate (reverse flow)
		gen, gen_res = vae.generate(frame_cond)
		logger.info(f"✓ Generate path:")
		logger.info(f"  - Generated mels: {gen.shape} [B, n_mels, T]")
		if gen_res is not None:
			logger.info(f"  - Generated residual: {gen_res.shape} [B, T, E]")
		
		# Flow reversibility check at latent resolution
		lat_cond = vae._align_and_downsample_cond(frame_cond)  # [B, T', C]
		b = int(lat_cond.shape[0])
		tp = int(lat_cond.shape[1])
		c = int(vae.model_channels)
		z0 = keras.random.normal((b, tp, c))
		zf = vae.flow(z0, cond=lat_cond, reverse=False)
		zr = vae.flow(zf, cond=lat_cond, reverse=True)
		diff = jnp.max(jnp.abs(zr - z0))
		logger.info(f"✓ Flow invertibility check: max|z - inv(flow(flow(z)))| = {float(diff):.6e}")
		
		return True
	except Exception as e:
		logger.error(f"✗ Forward test failed: {e}")
		raise


def main():
	parser = argparse.ArgumentParser(description="Validate VAE model structure and forward paths")
	parser.add_argument("--ljspeech_dir", type=str, default="data/LJSpeech-1.1")
	parser.add_argument("--alignments_dir", type=str, default="data/ljspeech_alignments/LJSpeech")
	parser.add_argument("--encoder_weights", type=str, default="")
	parser.add_argument("--n_mels", type=int, default=80)
	parser.add_argument("--embed_dim", type=int, default=256)
	args = parser.parse_args()
	
	logger.info("Validating VAE setup")
	try:
		ds = test_dataset(args.ljspeech_dir, args.alignments_dir)
		text_encoder, vae = test_models(ds, encoder_weights=args.encoder_weights, n_mels=args.n_mels, embed_dim=args.embed_dim)
		_ = test_forward_batch(ds, text_encoder, vae, batch_size=2)
		logger.info("\n" + "=" * 60)
		logger.info("✓ ALL TESTS PASSED! VAE structure is valid.")
		logger.info("=" * 60)
		logger.info("You can start training with:")
		logger.info("  uv run python scripts/train_vae.py --encoder_weights <path_to_encoder_weights>")
	except Exception as e:
		logger.info("\n" + "=" * 60)
		logger.info("✗ VALIDATION FAILED")
		logger.info("=" * 60)
		logger.error(f"Error: {e}")
		raise


if __name__ == "__main__":
	main()

"""Test script to verify VAE setup before training."""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import logging
import jax.numpy as jnp

from iris.encoder import PhonemeEncoder
from iris.vae import create_vae
from iris.datasets import LJSpeechVAEDataset, collate_vae_batch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("Testing VAE Setup")
    logger.info("=" * 70)
    
    # Load dataset (just first sample)
    logger.info("\n1. Loading dataset...")
    dataset = LJSpeechVAEDataset(
        ljspeech_dir='data/LJSpeech-1.1',
        alignments_dir='data/ljspeech_alignments/LJSpeech',
        split='train',
        val_split=0.05,
        max_frames=1000,
        cache_dir='outputs/vae/cache'
    )
    
    vocab_size = dataset.get_vocab_size()
    logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
    logger.info(f"✓ Vocabulary size: {vocab_size}")
    
    # Test single sample
    sample = dataset[0]
    logger.info(f"\n2. Sample 0:")
    logger.info(f"  - File ID: {sample['file_id']}")
    logger.info(f"  - Phoneme IDs: {sample['phoneme_ids'].shape}")
    logger.info(f"  - Mel-spec: {sample['mel_spec'].shape}")
    logger.info(f"  - Durations: {sample['durations'].shape}")
    
    # Test batching
    logger.info(f"\n3. Testing batch collation...")
    batch = [dataset[i] for i in range(4)]
    collated = collate_vae_batch(batch)
    logger.info(f"✓ Phoneme IDs: {collated['phoneme_ids'].shape}")
    logger.info(f"✓ Mel-specs: {collated['mel_specs'].shape}")
    logger.info(f"✓ Durations: {collated['durations'].shape}")
    
    # Build encoder
    logger.info(f"\n4. Building encoder...")
    encoder = PhonemeEncoder(
        vocab_size=vocab_size,
        embed_dim=256,
        num_blocks=4,
        num_heads=4
    )
    
    dummy_phonemes = jnp.ones((1, 10), dtype='int32')
    _ = encoder(dummy_phonemes, training=False)
    
    logger.info(f"  Loading weights...")
    encoder.load_weights('outputs/encoder/checkpoints/encoder_best.weights.h5')
    encoder.trainable = False
    logger.info(f"✓ Encoder loaded and frozen")
    
    # Build VAE
    logger.info(f"\n5. Building VAE...")
    vae = create_vae(
        n_mels=80,
        cond_dim=256,
        model_channels=256,
        num_wavenet_blocks=4,
        down_stages=2,
        flow_layers=2,
        flow_hidden=256,
    )
    _ = vae(
        mels_bt_f=jnp.ones((1, 80, 16), dtype="float32"),
        frame_text_cond=jnp.ones((1, 16, 256), dtype="float32"),
        training=False,
    )
    logger.info(f"✓ VAE built: {vae.count_params():,} parameters")
     
    # Test forward pass with real data
    logger.info(f"\n6. Testing forward pass...")
    phoneme_ids = jnp.array(collated['phoneme_ids'])
    mel_specs = jnp.array(collated['mel_specs'])
     
    # Get text conditioning
    text_cond = encoder(phoneme_ids, training=False)
    logger.info(f"✓ Text conditioning: {text_cond.shape}")
    
    # Build frame-level conditioning aligned to mel frames
    durations = np.array(collated['durations'])
    mel_lengths = np.array(collated['mel_lengths'])
    enc_out_np = np.array(text_cond)  # [B, P, E]
    frame_cond_np, mask_np = build_frame_level_condition(enc_out_np, durations, mel_lengths)
    t_max = frame_cond_np.shape[1]
    # Ensure mel T matches frame_cond T
    if mel_specs.shape[2] != t_max:
        if mel_specs.shape[2] > t_max:
            mel_specs = mel_specs[:, :, :t_max]
        else:
            pad = jnp.zeros((mel_specs.shape[0], mel_specs.shape[1], t_max - mel_specs.shape[2]), dtype=mel_specs.dtype)
            mel_specs = jnp.concatenate([mel_specs, pad], axis=2)
    frame_cond = jnp.array(frame_cond_np)
    mask_bt = jnp.array(mask_np)
    
    # VAE forward
    mel_recon, (mu, logvar), residual = vae(
        mel_specs,
        frame_cond,
        training=False
    )
    
    logger.info(f"✓ VAE forward pass successful:")
    logger.info(f"  - Mel reconstructed: {mel_recon.shape}")
    logger.info(f"  - Posterior mean/logvar: {mu.shape}, {logvar.shape}")
    if residual is not None:
        logger.info(f"  - Residual embedding: {residual.shape}")
    
    # Compute loss (L1 + KL)
    recon_loss = vae.compute_recon_l1(mel_specs, mel_recon, mask=mask_bt)
    kl_loss = vae.compute_kl(mu, logvar)
    logger.info(f"\n7. Loss computation:")
    logger.info(f"  - Reconstruction L1: {float(recon_loss):.4f}")
    logger.info(f"  - KL divergence: {float(kl_loss):.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ ALL TESTS PASSED!")
    logger.info("=" * 70)
    logger.info("Ready to train VAE with:")
    logger.info("  uv run python scripts/train_vae.py")
    logger.info("")


if __name__ == '__main__':
    main()

