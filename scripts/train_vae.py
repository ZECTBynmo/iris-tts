"""Training script for text-conditioned VAE using pre-trained encoder."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

import keras
from keras import ops
import jax.numpy as jnp
from typing import Optional

from iris.encoder import PhonemeEncoder
from iris.datasets import LJSpeechVAEDataset, collate_vae_batch
from iris.vae import TextConditionedVAE


# Setup logging
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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
		# For each frame t in [0, t_len), find phoneme index k s.t. t < cum[k]
		# Using searchsorted on cum
		indices = np.searchsorted(cum, np.arange(t_len), side="right")
		indices = np.clip(indices, 0, p_len - 1)
		enc_i = encoder_outputs[i]  # [P, E]
		frame_cond[i, :t_len] = enc_i[indices]
		mask[i, :t_len] = True
	
	return frame_cond, mask


class VAETrainer(keras.Model):
	"""Wraps VAE to define Keras losses for train_on_batch/test_on_batch."""
	def __init__(self, vae: TextConditionedVAE, kl_weight: float = 1e-3, **kwargs):
		super().__init__(**kwargs)
		self.vae = vae
		self.kl_weight = kl_weight
	
	def call(self, inputs, training=False):
		# inputs: (mels_bt_f, frame_text_cond, mask_bt)
		mels_bt_f, frame_text_cond, _ = inputs
		recon, (mean, logvar), residual = self.vae(mels_bt_f, frame_text_cond, training=training)
		return recon
	
	def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
		# x: (mels_bt_f, frame_text_cond, mask_bt)
		# y: target mels_bt_f
		mels_bt_f, frame_text_cond, mask_bt = x
		# Forward pass again to access posterior stats
		recon, (mean, logvar), _ = self.vae(mels_bt_f, frame_text_cond, training=True)
		# Compute mask-aware recon L1
		loss_recon = self.vae.compute_recon_l1(y, recon, mask=mask_bt)
		loss_kl = self.vae.compute_kl(mean, logvar)
		return loss_recon + self.kl_weight * loss_kl


def train_vae(
	ljspeech_dir: str,
	alignments_dir: str,
	output_dir: str,
	encoder_weights: Optional[str] = None,
	freeze_encoder: bool = True,
	n_mels: int = 80,
	embed_dim: int = 256,
	model_channels: int = 256,
	num_blocks: int = 6,
	down_stages: int = 2,
	flow_layers: int = 4,
	flow_hidden: int = 256,
	batch_size: int = 8,
	num_epochs: int = 50,
	learning_rate: float = 2e-4,
	val_split: float = 0.05,
	checkpoint_dir: str = "checkpoints_vae",
	log_interval: int = 100,
	kl_weight: float = 1e-3,
):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	ckpt_dir = output_dir / checkpoint_dir
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	
	# Save training config
	config = {
		"ljspeech_dir": ljspeech_dir,
		"alignments_dir": alignments_dir,
		"n_mels": n_mels,
		"embed_dim": embed_dim,
		"model_channels": model_channels,
		"num_blocks": num_blocks,
		"down_stages": down_stages,
		"flow_layers": flow_layers,
		"flow_hidden": flow_hidden,
		"batch_size": batch_size,
		"num_epochs": num_epochs,
		"learning_rate": learning_rate,
		"val_split": val_split,
		"freeze_encoder": freeze_encoder,
		"encoder_weights": encoder_weights,
		"kl_weight": kl_weight,
	}
	with open(output_dir / "config_vae.json", "w") as f:
		json.dump(config, f, indent=2)
	logger.info(f"Saved config to {output_dir / 'config_vae.json'}")
	
	# Datasets
	logger.info("Loading datasets...")
	train_dataset = LJSpeechVAEDataset(
		ljspeech_dir=ljspeech_dir,
		alignments_dir=alignments_dir,
		split="train",
		val_split=val_split,
	)
	val_dataset = LJSpeechVAEDataset(
		ljspeech_dir=ljspeech_dir,
		alignments_dir=alignments_dir,
		split="val",
		val_split=val_split,
		cache_dir=str(output_dir / "cache"),
	)
	vocab_size = train_dataset.get_vocab_size()
	logger.info(f"Vocabulary size: {vocab_size}")
	logger.info(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
	
	# Pre-trained text encoder
	logger.info("Building text encoder for conditioning...")
	text_encoder = PhonemeEncoder(
		vocab_size=vocab_size,
		embed_dim=embed_dim,
		num_blocks=4,
		num_heads=4,
		dropout=0.1,
	)
	# Build
	_ = text_encoder(jnp.ones((1, 8), dtype="int32"), training=False)
	if encoder_weights:
		enc_path = Path(encoder_weights)
		logger.info(f"Attempting to load encoder weights from: {enc_path.resolve()}")
		if enc_path.exists():
			logger.info(f"Loading encoder weights from {enc_path}")
			text_encoder.load_weights(str(enc_path))
		else:
			logger.warning(f"Encoder weights not found at {enc_path}")
	else:
		logger.warning("Encoder weights not provided or not found. Using randomly initialized encoder.")
	if freeze_encoder:
		text_encoder.trainable = False
	
	# VAE model
	logger.info("Building VAE...")
	vae = TextConditionedVAE(
		n_mels=n_mels,
		cond_dim=embed_dim,
		model_channels=model_channels,
		num_wavenet_blocks=num_blocks,
		down_stages=down_stages,
		flow_layers=flow_layers,
		flow_hidden=flow_hidden,
	)
	# Build with dummy input
	_ = vae(
		mels_bt_f=jnp.ones((1, n_mels, 16), dtype="float32"),
		frame_text_cond=jnp.ones((1, 16, embed_dim), dtype="float32"),
		training=False,
	)
	
	logger.info(f"VAE parameters: {vae.count_params():,}")
	
	# Training wrapper
	model = VAETrainer(vae=vae, kl_weight=kl_weight)
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
		loss=model.compute_loss,
		jit_compile=True,
	)
	logger.info("Compiled VAE trainer with JIT.")
	
	def iterate_batches(dataset, bs):
		num_batches = int(np.ceil(len(dataset) / bs))
		for step in range(num_batches):
			start = step * bs
			end = min(start + bs, len(dataset))
			yield [dataset[i] for i in range(start, end)]
	
	best_val = float("inf")
	global_step = 0
	
	for epoch in range(num_epochs):
		logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
		train_losses = []
		
		# Shuffle
		perm = np.random.permutation(len(train_dataset))
		
		num_train_batches = int(np.ceil(len(train_dataset) / batch_size))
		pbar = tqdm(range(0, len(train_dataset), batch_size), total=num_train_batches, desc=f"train {epoch+1}/{num_epochs}")
		for step, idxs in enumerate(pbar):
			batch_idxs = perm[idxs: idxs + batch_size]
			batch = [train_dataset[int(i)] for i in batch_idxs]
			col = collate_vae_batch(batch)
			
			# Prepare inputs
			phoneme_ids = jnp.array(col["phoneme_ids"])
			mels_bt_f = jnp.array(col["mel_specs"])
			durations = np.array(col["durations"])
			phon_len = np.array(col["phoneme_lengths"])
			mel_len = np.array(col["mel_lengths"])
			
			# Encoder outputs [B, P, E] using pre-trained encoder
			enc_out = text_encoder(phoneme_ids, training=False)  # jnp
			enc_out_np = np.array(enc_out)  # to numpy for CPU shaping
			
			# Frame-level conditioning aligned to mel frames
			frame_cond_np, mask_np = build_frame_level_condition(enc_out_np, durations, mel_len)
			
			# Round target length up to multiple of up/down factor to avoid dynamic slicing
			factor = 2 ** down_stages
			t_max = frame_cond_np.shape[1]
			target_len = int(np.ceil(t_max / factor) * factor)
			# Pad mel specs to target_len
			mels_bt_f_np = np.array(mels_bt_f)
			if mels_bt_f_np.shape[2] < target_len:
				pad = np.zeros((mels_bt_f_np.shape[0], mels_bt_f_np.shape[1], target_len - mels_bt_f_np.shape[2]), dtype=mels_bt_f_np.dtype)
				mels_bt_f_np = np.concatenate([mels_bt_f_np, pad], axis=2)
			elif mels_bt_f_np.shape[2] > target_len:
				mels_bt_f_np = mels_bt_f_np[:, :, :target_len]
			# Pad frame_cond and mask to target_len with zeros
			if frame_cond_np.shape[1] < target_len:
				pad = np.zeros((frame_cond_np.shape[0], target_len - frame_cond_np.shape[1], frame_cond_np.shape[2]), dtype=frame_cond_np.dtype)
				frame_cond_np = np.concatenate([frame_cond_np, pad], axis=1)
			if mask_np.shape[1] < target_len:
				padm = np.zeros((mask_np.shape[0], target_len - mask_np.shape[1]), dtype=mask_np.dtype)
				mask_np = np.concatenate([mask_np, padm], axis=1)
			
			# Convert to jnp
			frame_cond = jnp.array(frame_cond_np)
			mask_bt = jnp.array(mask_np)
			mels_bt_f = jnp.array(mels_bt_f_np)
			
			# Train step
			loss = model.train_on_batch(
				x=(mels_bt_f, frame_cond, mask_bt),
				y=mels_bt_f,
			)
			train_losses.append(float(loss))
			global_step += 1
			avg = np.mean(train_losses[-log_interval:]) if (len(train_losses) >= 1) else float(loss)
			pbar.set_postfix(loss=f"{float(loss):.4f}", avg=f"{avg:.4f}")
		
		train_loss = float(np.mean(train_losses)) if train_losses else 0.0
		
		# Validation
		val_losses = []
		num_val_batches = int(np.ceil(len(val_dataset) / batch_size))
		pbar_val = tqdm(range(0, len(val_dataset), batch_size), total=num_val_batches, desc=f"val   {epoch+1}/{num_epochs}")
		for start in pbar_val:
			end = min(start + batch_size, len(val_dataset))
			batch = [val_dataset[i] for i in range(start, end)]
			col = collate_vae_batch(batch)
			phoneme_ids = jnp.array(col["phoneme_ids"])
			mels_bt_f = jnp.array(col["mel_specs"])
			durations = np.array(col["durations"])
			mel_len = np.array(col["mel_lengths"])
			
			enc_out = text_encoder(phoneme_ids, training=False)
			enc_out_np = np.array(enc_out)
			frame_cond_np, mask_np = build_frame_level_condition(enc_out_np, durations, mel_len)
			# Round and pad to factor length
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
			
			loss = model.test_on_batch(
				x=(mels_bt_f, frame_cond, mask_bt),
				y=mels_bt_f,
			)
			val_losses.append(float(loss))
			pbar_val.set_postfix(loss=f"{float(loss):.4f}")
		
		val_loss = float(np.mean(val_losses)) if val_losses else 0.0
		logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
		
		# Checkpoint
		if val_loss < best_val:
			best_val = val_loss
			model.save_weights(ckpt_dir / "vae_best.weights.h5")
			vae.save_weights(ckpt_dir / "vae_core_best.weights.h5")
			logger.info(f"Saved best checkpoints (val {best_val:.4f})")
		
		if (epoch + 1) % 5 == 0:
			model.save_weights(ckpt_dir / f"vae_epoch_{epoch+1}.weights.h5")
			logger.info(f"Saved periodic checkpoint at epoch {epoch+1}")
	
	# Final save
	model.save_weights(ckpt_dir / "vae_final.weights.h5")
	vae.save_weights(ckpt_dir / "vae_core_final.weights.h5")
	logger.info("Training complete.")


def main():
	parser = argparse.ArgumentParser(description="Train PortaSpeech-style text-conditioned VAE")
	parser.add_argument("--ljspeech_dir", type=str, default="data/LJSpeech-1.1")
	parser.add_argument("--alignments_dir", type=str, default="data/ljspeech_alignments/LJSpeech")
	parser.add_argument("--output_dir", type=str, default="outputs/vae")
	parser.add_argument("--encoder_weights", type=str, default="outputs/encoder/checkpoints/encoder_best.weights.h5")
	parser.add_argument("--freeze_encoder", action="store_true")
	parser.add_argument("--n_mels", type=int, default=80)
	parser.add_argument("--embed_dim", type=int, default=256)
	parser.add_argument("--model_channels", type=int, default=256)
	parser.add_argument("--num_blocks", type=int, default=6)
	parser.add_argument("--down_stages", type=int, default=2)
	parser.add_argument("--flow_layers", type=int, default=4)
	parser.add_argument("--flow_hidden", type=int, default=256)
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--num_epochs", type=int, default=50)
	parser.add_argument("--learning_rate", type=float, default=2e-4)
	parser.add_argument("--val_split", type=float, default=0.05)
	parser.add_argument("--kl_weight", type=float, default=1e-3)
	parser.add_argument("--log_interval", type=int, default=100)
	args = parser.parse_args()
	
	train_vae(
		ljspeech_dir=args.ljspeech_dir,
		alignments_dir=args.alignments_dir,
		output_dir=args.output_dir,
		encoder_weights=args.encoder_weights if args.encoder_weights else None,
		freeze_encoder=args.freeze_encoder,
		n_mels=args.n_mels,
		embed_dim=args.embed_dim,
		model_channels=args.model_channels,
		num_blocks=args.num_blocks,
		down_stages=args.down_stages,
		flow_layers=args.flow_layers,
		flow_hidden=args.flow_hidden,
		batch_size=args.batch_size,
		num_epochs=args.num_epochs,
		learning_rate=args.learning_rate,
		val_split=args.val_split,
		log_interval=args.log_interval,
		kl_weight=args.kl_weight,
	)


if __name__ == "__main__":
	main()


