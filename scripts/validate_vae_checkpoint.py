"""Validate saved VAE checkpoints by running reconstruction on validation batches."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import json
from pathlib import Path
import logging
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

import keras

from iris.encoder import PhonemeEncoder
from iris.vae import TextConditionedVAE
from iris.postnet import PostNet
from iris.datasets import LJSpeechVAEDataset, collate_vae_batch


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_frame_level_condition(encoder_outputs, durations, mel_lengths):
	"""Map phoneme-level encoder outputs to frame-level by durations."""
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


def main():
	parser = argparse.ArgumentParser(description="Validate VAE checkpoint on validation split")
	parser.add_argument("--ljspeech_dir", type=str, default="data/LJSpeech-1.1")
	parser.add_argument("--alignments_dir", type=str, default="data/ljspeech_alignments/LJSpeech")
	parser.add_argument("--output_dir", type=str, default="outputs/vae")
	parser.add_argument("--encoder_weights", type=str, default="outputs/encoder/checkpoints/encoder_best.weights.h5")
	parser.add_argument("--vae_weights", type=str, default="outputs/vae/checkpoints_vae/vae_core_best.weights.h5")
	parser.add_argument("--config", type=str, default="outputs/vae/config_vae.json")
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--num_batches", type=int, default=4)
	args = parser.parse_args()
	
	# Load config if available
	cfg_path = Path(args.config)
	if cfg_path.exists():
		with open(cfg_path, "r") as f:
			cfg = json.load(f)
		n_mels = cfg.get("n_mels", 80)
		embed_dim = cfg.get("embed_dim", 256)
		model_channels = cfg.get("model_channels", 256)
		num_blocks = cfg.get("num_blocks", 6)
		down_stages = cfg.get("down_stages", 2)
		flow_layers = cfg.get("flow_layers", 4)
		flow_hidden = cfg.get("flow_hidden", 256)
		kl_weight = cfg.get("kl_weight", 1e-3)
		logger.info(f"Loaded VAE config from {cfg_path}")
	else:
		# Fallback to defaults
		n_mels = 80
		embed_dim = 256
		model_channels = 256
		num_blocks = 6
		down_stages = 2
		flow_layers = 4
		flow_hidden = 256
		kl_weight = 1e-3
		logger.warning(f"Config not found at {cfg_path}, using defaults")
	
	# Datasets (use validation split)
	logger.info("Loading validation dataset...")
	val_dataset = LJSpeechVAEDataset(
		ljspeech_dir=args.ljspeech_dir,
		alignments_dir=args.alignments_dir,
		split="val",
		val_split=0.05,
		cache_dir=str(Path(args.output_dir) / "cache"),
	)
	vocab_size = val_dataset.get_vocab_size()
	logger.info(f"Validation samples: {len(val_dataset)} | Vocab size: {vocab_size}")
	
	# Build text encoder and load weights
	logger.info("Building text encoder...")
	text_encoder = PhonemeEncoder(
		vocab_size=vocab_size,
		embed_dim=embed_dim,
		num_blocks=4,
		num_heads=4,
		dropout=0.1,
	)
	_ = text_encoder(jnp.ones((1, 8), dtype="int32"), training=False)
	enc_path = Path(args.encoder_weights)
	logger.info(f"Loading encoder weights: {enc_path.resolve()}")
	text_encoder.load_weights(str(enc_path))
	text_encoder.trainable = False
	
	# Build VAE and load weights
	logger.info("Building VAE and loading checkpoint...")
	vae = TextConditionedVAE(
		n_mels=n_mels,
		cond_dim=embed_dim,
		model_channels=model_channels,
		num_wavenet_blocks=num_blocks,
		down_stages=down_stages,
		flow_layers=flow_layers,
		flow_hidden=flow_hidden,
	)
	# Warmup
	_ = vae(
		mels_bt_f=jnp.ones((1, n_mels, 16), dtype="float32"),
		frame_text_cond=jnp.ones((1, 16, embed_dim), dtype="float32"),
		training=False,
	)
	vae_path = Path(args.vae_weights)
	logger.info(f"Loading VAE weights: {vae_path.resolve()}")
	vae.load_weights(str(vae_path))

	# Optional PostNet
	postnet = None
	postnet_path = Path(args.output_dir) / "checkpoints_vae" / "postnet_best.weights.h5"
	if postnet_path.exists():
		logger.info(f"Loading PostNet weights: {postnet_path.resolve()}")
		postnet = PostNet(n_mels=n_mels)
		_ = postnet(jnp.ones((1, n_mels, 16), dtype="float32"), training=False)
		postnet.load_weights(str(postnet_path))

	# Validate a few batches
	recon_losses = []
	kl_losses = []
	comp_losses = []
	num_iters = min(args.num_batches, int(np.ceil(len(val_dataset) / args.batch_size)))
	pbar = tqdm(range(num_iters), desc="validating")
	
	start_idx = 0
	for _ in pbar:
		end_idx = min(start_idx + args.batch_size, len(val_dataset))
		batch = [val_dataset[i] for i in range(start_idx, end_idx)]
		start_idx = end_idx
		if not batch:
			break
		
		col = collate_vae_batch(batch)
		phoneme_ids = jnp.array(col["phoneme_ids"])
		mels_bt_f = jnp.array(col["mel_specs"])
		durations = np.array(col["durations"])
		mel_len = np.array(col["mel_lengths"])
		
		# Encoder conditioning
		enc_out = text_encoder(phoneme_ids, training=False)
		enc_out_np = np.array(enc_out)
		frame_cond_np, mask_np = build_frame_level_condition(enc_out_np, durations, mel_len)
		
		# Round to factor and pad
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
		
		# Forward and metrics
		recon, (mean, logvar), residual = vae(mels_bt_f, frame_cond, training=False)
		if postnet is not None:
			refined = postnet(recon, training=False)
			l1 = vae.compute_recon_l1(mels_bt_f, refined, mask=mask_bt)
		else:
			l1 = vae.compute_recon_l1(mels_bt_f, recon, mask=mask_bt)
		kl = vae.compute_kl(mean, logvar)
		comp = l1 + kl_weight * kl
		recon_losses.append(float(l1))
		kl_losses.append(float(kl))
		comp_losses.append(float(comp))
		pbar.set_postfix(recon=f"{float(l1):.4f}", kl=f"{float(kl):.4f}", comp=f"{float(comp):.4f}")
		
	# Summary
	if recon_losses:
		logger.info(f"Validation summary over {len(recon_losses)} batches:")
		logger.info(f"  Recon L1: mean={np.mean(recon_losses):.4f}, std={np.std(recon_losses):.4f}")
		logger.info(f"  KL:       mean={np.mean(kl_losses):.4f}, std={np.std(kl_losses):.4f}")
		logger.info(f"  Composite (L1 + {kl_weight:.1e}*KL): mean={np.mean(comp_losses):.4f}, std={np.std(comp_losses):.4f}")
	else:
		logger.warning("No validation batches processed.")
	
	# Quick generate test
	try:
		logger.info("Testing generate() on a small slice...")
		# Use first batch again if available
		idxs = list(range(min(args.batch_size, len(val_dataset))))
		batch = [val_dataset[i] for i in idxs]
		col = collate_vae_batch(batch)
		phoneme_ids = jnp.array(col["phoneme_ids"])
		durations = np.array(col["durations"])
		mel_len = np.array(col["mel_lengths"])
		enc_out = text_encoder(phoneme_ids, training=False)
		enc_out_np = np.array(enc_out)
		frame_cond_np, _ = build_frame_level_condition(enc_out_np, durations, mel_len)
		# Pad to factor
		t_max = frame_cond_np.shape[1]
		target_len = int(np.ceil(t_max / (2 ** down_stages)) * (2 ** down_stages))
		if frame_cond_np.shape[1] < target_len:
			pad = np.zeros((frame_cond_np.shape[0], target_len - frame_cond_np.shape[1], frame_cond_np.shape[2]), dtype=frame_cond_np.dtype)
			frame_cond_np = np.concatenate([frame_cond_np, pad], axis=1)
		frame_cond = jnp.array(frame_cond_np)
		gen_mels, gen_res = vae.generate(frame_cond)
		if postnet is not None:
			gen_mels = postnet(gen_mels, training=False)
		logger.info(f"  generate(): mel {gen_mels.shape}, residual {gen_res.shape if gen_res is not None else None}")
	except Exception as e:
		logger.warning(f"generate() test failed: {e}")


if __name__ == "__main__":
	main()


