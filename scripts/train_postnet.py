"""Train Tacotron2-style PostNet on top of a pre-trained VAE."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import logging
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

import keras
import jax.numpy as jnp

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


class PostNetTrainer(keras.Model):
	"""Wraps VAE + PostNet; computes L1 on refined mel (mask-aware)."""
	def __init__(self, vae: TextConditionedVAE, postnet: PostNet, **kwargs):
		super().__init__(**kwargs)
		self.vae = vae
		self.postnet = postnet
	
	def call(self, inputs, training=False):
		mels_bt_f, frame_text_cond, _ = inputs
		recon, _, _ = self.vae(mels_bt_f, frame_text_cond, training=False)  # VAE forward (typically frozen)
		refined = self.postnet(recon, training=training)
		return refined
	
	def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
		mels_bt_f, frame_text_cond, mask_bt = x
		recon, _, _ = self.vae(mels_bt_f, frame_text_cond, training=False)
		refined = self.postnet(recon, training=True)
		# L1 with mask
		diff = keras.ops.abs(y - refined)
		if mask_bt is not None:
			m = keras.ops.expand_dims(mask_bt, axis=1)  # [B, 1, T]
			diff = diff * m
			return keras.ops.sum(diff) / (keras.ops.sum(m) * keras.ops.cast(keras.ops.shape(diff)[1], diff.dtype) + 1e-6)
		return keras.ops.mean(diff)


def train_postnet(
	ljspeech_dir: str,
	alignments_dir: str,
	output_dir: str,
	encoder_weights: str,
	vae_core_weights: str,
	freeze_vae: bool = True,
	batch_size: int = 16,
	num_epochs: int = 30,
	learning_rate: float = 1e-4,
	val_split: float = 0.05,
	log_interval: int = 100,
	postnet_layers: int = 5,
	postnet_channels: int = 512,
	postnet_dropout: float = 0.5,
):
	output_dir = Path(output_dir)
	ckpt_dir = output_dir / "checkpoints_postnet"
	ckpt_dir.mkdir(parents=True, exist_ok=True)
	
	# Load config to match VAE architecture
	cfg_path = output_dir / "config_vae.json"
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
	else:
		logger.warning(f"No VAE config found at {cfg_path}, using defaults.")
		n_mels = 80
		embed_dim = 256
		model_channels = 256
		num_blocks = 6
		down_stages = 2
		flow_layers = 4
		flow_hidden = 256
	
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
	logger.info(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
	
	# Text encoder
	logger.info("Building text encoder...")
	text_encoder = PhonemeEncoder(
		vocab_size=vocab_size,
		embed_dim=embed_dim,
		num_blocks=4,
		num_heads=4,
		dropout=0.1,
	)
	_ = text_encoder(jnp.ones((1, 8), dtype="int32"), training=False)
	logger.info(f"Loading encoder weights: {Path(encoder_weights).resolve()}")
	text_encoder.load_weights(encoder_weights)
	text_encoder.trainable = False
	
	# VAE
	logger.info("Building VAE and loading core weights...")
	vae = TextConditionedVAE(
		n_mels=n_mels,
		cond_dim=embed_dim,
		model_channels=model_channels,
		num_wavenet_blocks=num_blocks,
		down_stages=down_stages,
		flow_layers=flow_layers,
		flow_hidden=flow_hidden,
	)
	_ = vae(
		mels_bt_f=jnp.ones((1, n_mels, 16), dtype="float32"),
		frame_text_cond=jnp.ones((1, 16, embed_dim), dtype="float32"),
		training=False,
	)
	logger.info(f"Loading VAE core: {Path(vae_core_weights).resolve()}")
	vae.load_weights(vae_core_weights)
	if freeze_vae:
		vae.trainable = False
	
	# PostNet
	logger.info("Building PostNet...")
	postnet = PostNet(
		n_mels=n_mels,
		num_layers=postnet_layers,
		channels=postnet_channels,
		dropout=postnet_dropout,
	)
	_ = postnet(jnp.ones((1, n_mels, 16), dtype="float32"), training=False)
	logger.info(f"PostNet parameters: {postnet.count_params():,}")
	
	# Trainer
	model = PostNetTrainer(vae=vae, postnet=postnet)
	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
		loss=model.compute_loss,
		jit_compile=True,
	)
	
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
		perm = np.random.permutation(len(train_dataset))
		
		num_train_batches = int(np.ceil(len(train_dataset) / batch_size))
		pbar = tqdm(range(0, len(train_dataset), batch_size), total=num_train_batches, desc=f"train {epoch+1}/{num_epochs}")
		for step, idxs in enumerate(pbar):
			batch_idxs = perm[idxs: idxs + batch_size]
			batch = [train_dataset[int(i)] for i in batch_idxs]
			col = collate_vae_batch(batch)
			
			phoneme_ids = jnp.array(col["phoneme_ids"])
			mels_bt_f = jnp.array(col["mel_specs"])
			durations = np.array(col["durations"])
			mel_len = np.array(col["mel_lengths"])
			
			# Conditioning
			enc_out = text_encoder(phoneme_ids, training=False)
			enc_out_np = np.array(enc_out)
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
			
			# Train
			loss = model.train_on_batch(
				x=(mels_bt_f, frame_cond, mask_bt),
				y=mels_bt_f,
			)
			train_losses.append(float(loss))
			global_step += 1
			avg = np.mean(train_losses[-log_interval:]) if train_losses else float(loss)
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
		logger.info(f"Epoch {epoch+1} | Train L1: {train_loss:.4f} | Val L1: {val_loss:.4f}")
		
		# Checkpoint
		if val_loss < best_val:
			best_val = val_loss
			model.save_weights(ckpt_dir / "postnet_best_wrapper.weights.h5")
			postnet.save_weights(ckpt_dir / "postnet_best.weights.h5")
			logger.info(f"Saved best PostNet (val {best_val:.4f})")
		
		if (epoch + 1) % 5 == 0:
			postnet.save_weights(ckpt_dir / f"postnet_epoch_{epoch+1}.weights.h5")
			logger.info(f"Saved periodic PostNet at epoch {epoch+1}")
	
	postnet.save_weights(ckpt_dir / "postnet_final.weights.h5")
	logger.info("PostNet training complete.")


def main():
	parser = argparse.ArgumentParser(description="Train PostNet on top of a pre-trained VAE")
	parser.add_argument("--ljspeech_dir", type=str, default="data/LJSpeech-1.1")
	parser.add_argument("--alignments_dir", type=str, default="data/ljspeech_alignments/LJSpeech")
	parser.add_argument("--output_dir", type=str, default="outputs/vae")
	parser.add_argument("--encoder_weights", type=str, default="outputs/encoder/checkpoints/encoder_best.weights.h5")
	parser.add_argument("--vae_core_weights", type=str, default="outputs/vae/checkpoints_vae/vae_core_best.weights.h5")
	parser.add_argument("--freeze_vae", action="store_true")
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--num_epochs", type=int, default=30)
	parser.add_argument("--learning_rate", type=float, default=1e-4)
	parser.add_argument("--val_split", type=float, default=0.05)
	parser.add_argument("--log_interval", type=int, default=100)
	parser.add_argument("--postnet_layers", type=int, default=4)
	parser.add_argument("--postnet_channels", type=int, default=256)
	parser.add_argument("--postnet_dropout", type=float, default=0.5)
	args = parser.parse_args()
	
	train_postnet(
		ljspeech_dir=args.ljspeech_dir,
		alignments_dir=args.alignments_dir,
		output_dir=args.output_dir,
		encoder_weights=args.encoder_weights,
		vae_core_weights=args.vae_core_weights,
		freeze_vae=args.freeze_vae,
		batch_size=args.batch_size,
		num_epochs=args.num_epochs,
		learning_rate=args.learning_rate,
		val_split=args.val_split,
		log_interval=args.log_interval,
		postnet_layers=args.postnet_layers,
		postnet_channels=args.postnet_channels,
		postnet_dropout=args.postnet_dropout,
	)


if __name__ == "__main__":
	main()


