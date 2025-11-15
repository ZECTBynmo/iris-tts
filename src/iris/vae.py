"""Text-conditioned VAE with VPFlow, FiLM-WaveNet blocks, and temporal down/up-sampling."""

import keras
from keras import layers, ops
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List


class FiLM(layers.Layer):
	"""Feature-wise Linear Modulation from a conditioning tensor.
	
	Computes per-channel scale and shift from conditioning and applies:
	  y = gamma * x + beta
	"""
	def __init__(self, channels: int, name: Optional[str] = None):
		super().__init__(name=name)
		self.channels = channels
		self.proj = layers.Dense(2 * channels)
	
	def call(self, x, cond):
		"""
		Args:
			x: [batch, time, channels]
			cond: [batch, time, cond_dim] (time-aligned to x)
		"""
		gamma_beta = self.proj(cond)  # [B, T, 2C]
		gamma, beta = ops.split(gamma_beta, 2, axis=-1)
		return gamma * x + beta
	
	def get_config(self):
		cfg = super().get_config()
		cfg.update({"channels": self.channels})
		return cfg


class WaveNetResBlock(layers.Layer):
	"""Dilated Conv1D residual block with FiLM conditioning."""
	def __init__(self, channels: int, kernel_size: int, dilation_rate: int, dropout: float = 0.0, name: Optional[str] = None):
		super().__init__(name=name)
		self.channels = channels
		self.kernel_size = kernel_size
		self.dilation_rate = dilation_rate
		self.dropout_rate = dropout
		
		self.conv = layers.Conv1D(
			filters=channels,
			kernel_size=kernel_size,
			padding="same",
			dilation_rate=dilation_rate,
		)
		# No LayerNorm - following PortaSpeech NonCausalWaveNet
		self.film = FiLM(channels)
		self.dropout = layers.Dropout(dropout)
		self.res_proj = layers.Conv1D(filters=channels, kernel_size=1)
	
	def call(self, x, cond, training=False):
		"""
		Args:
			x: [B, T, C]
			cond: [B, T, Cc] (time-aligned conditioning)
		"""
		h = self.conv(x)
		h = ops.gelu(h)  # Activation before FiLM
		h = self.film(h, cond)
		h = self.dropout(h, training=training)
		return x + self.res_proj(h)
	
	def get_config(self):
		cfg = super().get_config()
		cfg.update({
			"channels": self.channels,
			"kernel_size": self.kernel_size,
			"dilation_rate": self.dilation_rate,
			"dropout": self.dropout_rate,
		})
		return cfg


class TemporalDownsample(layers.Layer):
	"""Stack of strided Conv1D layers to reduce time resolution."""
	def __init__(self, channels: int, num_stages: int = 2, kernel_size: int = 5, name: Optional[str] = None):
		super().__init__(name=name)
		self.channels = channels
		self.num_stages = num_stages
		self.kernel_size = kernel_size
		
		self.blocks = []
		for i in range(num_stages):
			self.blocks.append(
				layers.Conv1D(filters=channels, kernel_size=kernel_size, strides=2, padding="same")
			)
	
	def call(self, x, training=False):
		"""
		Args:
			x: [B, T, C_in]
		Returns:
			h: [B, T / 2^S, channels]
		"""
		h = x
		for conv in self.blocks:
			h = conv(h)
			h = ops.gelu(h)
		return h
	
	def get_downsample_factor(self) -> int:
		return 2 ** self.num_stages
	
	def get_config(self):
		cfg = super().get_config()
		cfg.update({
			"channels": self.channels,
			"num_stages": self.num_stages,
			"kernel_size": self.kernel_size,
		})
		return cfg


class TemporalUpsample(layers.Layer):
	"""Nearest-neighbor upsample + Conv1D refinement, repeated num_stages times."""
	def __init__(self, channels: int, num_stages: int = 2, kernel_size: int = 5, name: Optional[str] = None):
		super().__init__(name=name)
		self.channels = channels
		self.num_stages = num_stages
		self.kernel_size = kernel_size
		
		self.refine = []
		for i in range(num_stages):
			self.refine.append(
				layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same")
			)
	
	def _upsample2x(self, x):
		# x: [B, T, C] -> [B, 2T, C] by repeat
		b, t, c = x.shape
		x = ops.reshape(x, (b, t, 1, c))
		x = ops.repeat(x, 2, axis=2)
		return ops.reshape(x, (b, t * 2, c))
	
	def call(self, x, training=False):
		h = x
		for conv in self.refine:
			h = self._upsample2x(h)
			h = conv(h)
			h = ops.gelu(h)
		return h
	
	def get_upsample_factor(self) -> int:
		return 2 ** self.num_stages
	
	def get_config(self):
		cfg = super().get_config()
		cfg.update({
			"channels": self.channels,
			"num_stages": self.num_stages,
			"kernel_size": self.kernel_size,
		})
		return cfg


class APCoupling(layers.Layer):
	"""Additive coupling layer (volume-preserving) with FiLM conditioning."""
	def __init__(self, channels: int, hidden_channels: int, name: Optional[str] = None):
		super().__init__(name=name)
		self.channels = channels
		self.hidden_channels = hidden_channels
		# Conditioning projection - MUST be stored as self.xxx
		self.cond_proj = layers.Dense(channels // 2)
		# translation network t(x1, cond) - no LayerNorm (following PortaSpeech)
		self.net_pre = layers.Conv1D(filters=hidden_channels, kernel_size=3, padding="same")
		self.net_post = layers.Conv1D(
			filters=channels // 2, 
			kernel_size=1, 
			padding="same",
			kernel_initializer="zeros",  # PortaSpeech initializes to zero!
			bias_initializer="zeros",
		)
		# FiLM modulates the network output with conditioning
		self.film = FiLM(channels // 2)
	
	def call(self, x, cond, reverse: bool = False):
		"""
		Args:
			x: [B, T, C] where C is even
			cond: [B, T, Cc]
			reverse: apply inverse transformation if True
		"""
		c = x.shape[-1]
		x1, x2 = ops.split(x, 2, axis=-1)
		# Project conditioning to match x1 channels
		cond_embed = self.cond_proj(cond)
		cond_embed = ops.gelu(cond_embed)
		# Residual: combine x1 with conditioning
		h = x1 + cond_embed
		# Run through translation network
		h = self.net_pre(h)
		h = ops.gelu(h)
		t = self.net_post(h)  # Zero-initialized for stability
		# Apply FiLM to translation output using conditioning
		t = self.film(t, cond_embed)
		# Apply coupling
		if reverse:
			y2 = x2 - t
		else:
			y2 = x2 + t
		y = ops.concatenate([x1, y2], axis=-1)
		return y
	
	def get_config(self):
		cfg = super().get_config()
		cfg.update({
			"channels": self.channels,
			"hidden_channels": self.hidden_channels,
		})
		return cfg


class VolumePreservingFlow(layers.Layer):
	"""Stack of additive coupling layers (log-det = 0)."""
	def __init__(self, channels: int, num_layers: int = 4, hidden_channels: int = 256, name: Optional[str] = None):
		super().__init__(name=name)
		assert channels % 2 == 0, "Flow channels must be even."
		self.channels = channels
		self.num_layers = num_layers
		self.hidden_channels = hidden_channels
		self.layers_list = [APCoupling(channels=channels, hidden_channels=hidden_channels, name=f"ap_{i}") for i in range(num_layers)]
	
	def call(self, x, cond, reverse: bool = False):
		"""
		Args:
			x: [B, T, C]
			cond: [B, T, Cc] conditioning (time-aligned)
			reverse: if True, apply inverse (for inference)
		"""
		h = x
		if reverse:
			for layer in reversed(self.layers_list):
				h = layer(h, cond, reverse=True)
		else:
			for layer in self.layers_list:
				h = layer(h, cond, reverse=False)
		return h
	
	def get_config(self):
		cfg = super().get_config()
		cfg.update({
			"channels": self.channels,
			"num_layers": self.num_layers,
			"hidden_channels": self.hidden_channels,
		})
		return cfg


class TextConditionedVAE(keras.Model):
	"""
	PortaSpeech-style text-conditioned VAE:
	- Triple conditioning: encoder, flow, decoder all modulated by frame-level text features
	- Temporal downsampling before latent
	- Volume-preserving flow to avoid posterior collapse
	- WaveNet-style residual blocks with FiLM
	"""
	def __init__(
		self,
		n_mels: int,
		cond_dim: int,
		model_channels: int = 192,
		latent_dim: int = 16,
		num_wavenet_blocks: int = 8,
		decoder_blocks: int = 4,
		wavenet_kernel_size: int = 5,
		down_stages: int = 2,
		flow_layers: int = 4,
		flow_hidden: int = 64,
		dropout: float = 0.1,
		name: Optional[str] = None,
	):
		super().__init__(name=name)
		self.n_mels = n_mels
		self.cond_dim = cond_dim
		self.model_channels = model_channels
		self.latent_dim = latent_dim
		self.num_wavenet_blocks = num_wavenet_blocks
		self.decoder_blocks = decoder_blocks
		self.wavenet_kernel_size = wavenet_kernel_size
		self.down_stages = down_stages
		self.dropout_rate = dropout
		
		dec_channels = model_channels
		
		# RNG for JAX: use a SeedGenerator and pass it to random ops
		self.seed_generator = keras.random.SeedGenerator(1337)
		
		# Input projection (mel -> model_channels)
		# No input normalization - process raw mels directly (following PortaSpeech)
		self.in_proj = layers.Conv1D(filters=model_channels, kernel_size=1, padding="same")
		
		# WaveNet encoder blocks (pre-downsampling), each FiLM conditioned on frame-level text
		self.enc_blocks = [
			WaveNetResBlock(
				channels=model_channels,
				kernel_size=wavenet_kernel_size,
				dilation_rate=2 ** (i % 4),
				dropout=dropout,
				name=f"enc_block_{i}",
			)
			for i in range(num_wavenet_blocks)
		]
		
		# Downsample and mirror conditioning
		self.downsample = TemporalDownsample(channels=model_channels, num_stages=down_stages, kernel_size=5, name="downsample")
		self.down_cond_proj = layers.Conv1D(filters=model_channels, kernel_size=1, padding="same")
		
		# Latent projection (following PortaSpeech: encoder_decoder_hidden -> latent_hidden * 2)
		self.latent_enc_proj = layers.Dense(latent_dim * 2)  # Outputs mean + logvar
		
		# Flow in latent space (operates on latent_dim, not model_channels!)
		self.flow = VolumePreservingFlow(channels=latent_dim, num_layers=flow_layers, hidden_channels=flow_hidden, name="vpflow")
		
		# Decoder latent projection (latent_dim -> model_channels)
		self.latent_dec_proj = layers.Dense(dec_channels)
		
		# Decoder path (fewer blocks than encoder, following PortaSpeech)
		self.dec_blocks = [
			WaveNetResBlock(
				channels=dec_channels,
				kernel_size=wavenet_kernel_size,
				dilation_rate=2 ** (i % 4),
				dropout=dropout,
				name=f"dec_block_{i}",
			)
			for i in range(decoder_blocks)
		]
		self.upsample = TemporalUpsample(channels=dec_channels, num_stages=down_stages, kernel_size=5, name="upsample")
		# No normalization on output - let model learn natural mel range
		# Following PortaSpeech: layer_norm=False on decoder output
		self.out_proj = layers.Conv1D(filters=n_mels, kernel_size=1, padding="same")
		
		# Optional residual back to text encoder space: aggregate frame latents to phoneme-level outside this model if needed
		self.return_residual = True
		self.residual_proj = layers.Dense(cond_dim)
	
	def reparameterize(self, mean, logvar, training=False):
		if training:
			eps = keras.random.normal(shape=ops.shape(mean), seed=self.seed_generator)
			return mean + ops.exp(0.5 * logvar) * eps
		else:
			return mean
	
	def _align_and_downsample_cond(self, frame_cond):
		# frame_cond: [B, T, cond_dim] -> project and downsample to latent T'
		h = self.down_cond_proj(frame_cond)
		h = self.downsample(h)
		return h
	
	def call(
		self,
		mels_bt_f: jnp.ndarray,
		frame_text_cond: jnp.ndarray,
		training: bool = False,
	):
		"""
		Args:
			mels_bt_f: [B, n_mels, T] mel-spectrograms
			frame_text_cond: [B, T, cond_dim] frame-aligned text conditioning
		Returns:
			recon_mels: [B, n_mels, T]
			posterior_stats: (mean, logvar)
			optional residual embedding per frame: [B, T, cond_dim]
		"""
		# Convert mels to [B, T, n_mels]
		mels = ops.transpose(mels_bt_f, (0, 2, 1))
		
		# Direct projection - no input normalization (following PortaSpeech)
		h = self.in_proj(mels)
		
		# WaveNet encoder with FiLM conditioning (frame-level)
		for block in self.enc_blocks:
			h = block(h, frame_text_cond, training=training)
		
		# Downsample both h and conditioning for latent space
		lat_cond = self._align_and_downsample_cond(frame_text_cond)  # [B, T', C]
		lat_h = self.downsample(h)  # [B, T', C]
		
		# Project to latent space (following PortaSpeech: hidden -> latent_dim * 2)
		latent_params = self.latent_enc_proj(lat_h)  # [B, T', latent_dim*2]
		mean, logvar = ops.split(latent_params, 2, axis=-1)  # Each [B, T', latent_dim]
		z = self.reparameterize(mean, logvar, training=training)
		
		# Flow forward during training, reverse during inference handled by separate method
		z_flow = self.flow(z, cond=lat_cond, reverse=False)
		
		# Project latent back to hidden dimension (following PortaSpeech)
		d = self.latent_dec_proj(z_flow)  # [B, T', dec_channels]
		# First apply decoder blocks at downsampled resolution with downsampled conditioning
		for block in self.dec_blocks:
			d = block(d, lat_cond, training=training)
		# Upsample to full frame rate
		d_up = self.upsample(d)
		
		# NOTE: We assume input dimensions are pre-padded correctly in training
		# so d_up and frame_text_cond should already match.
		# The training script pads both mels and conditioning to the same target_len.
		
		# Direct output projection - no normalization (following PortaSpeech)
		out = self.out_proj(d_up)  # [B, T, n_mels]
		recon = ops.transpose(out, (0, 2, 1))  # [B, n_mels, T]
		
		# Frame-level residual embedding mapped to cond_dim (optional)
		residual = self.residual_proj(d_up) if self.return_residual else None
		
		return recon, (mean, logvar), residual
	
	def compute_kl(self, mean, logvar, mask: Optional[jnp.ndarray] = None):
		# KL between N(mean, exp(logvar)) and N(0, I), averaged over time and channels
		kl = -0.5 * (1 + logvar - ops.square(mean) - ops.exp(logvar))
		
		if mask is not None:
			# mask: [B, T] -> expand to channel dim
			m = ops.expand_dims(mask, axis=-1)  # [B, T, 1]
			kl = kl * m
			return ops.sum(kl) / (ops.sum(m) + 1e-8)
		return ops.mean(kl)
	
	def compute_recon_l1(self, target, recon, mask: Optional[jnp.ndarray] = None):
		# target/recon: [B, n_mels, T]
		diff = ops.abs(target - recon)
		if mask is not None:
			# mask: [B, T] -> expand to mel dims
			m = ops.expand_dims(mask, axis=1)  # [B, 1, T]
			diff = diff * m
			return ops.sum(diff) / (ops.sum(m) * ops.cast(ops.shape(diff)[1], diff.dtype) + 1e-6)
		return ops.mean(diff)
	
	def generate(self, frame_text_cond: jnp.ndarray, z_prior: Optional[jnp.ndarray] = None):
		"""Inference: sample from prior, apply inverse flow, and decode.
		
		Args:
			frame_text_cond: [B, T, cond_dim] frame-level text conditioning
			z_prior: Optional latent sample [B, T', C]. If None, sample standard normal with matching shapes.
		Returns:
			recon_mels: [B, n_mels, T]
			residual: [B, T, cond_dim]
		"""
		# Build latent conditioning by downsampling
		lat_cond = self._align_and_downsample_cond(frame_text_cond)
		b = ops.shape(lat_cond)[0]
		tp = ops.shape(lat_cond)[1]
		c = self.latent_dim  # Use latent_dim, not model_channels!
		if z_prior is None:
			z_prior = keras.random.normal((b, tp, c), seed=self.seed_generator)
		# Inverse flow
		z = self.flow(z_prior, cond=lat_cond, reverse=True)
		# Project latent to hidden dimension
		d = self.latent_dec_proj(z)
		for block in self.dec_blocks:
			d = block(d, lat_cond, training=False)
		# Upsample to full frame rate
		d_up = self.upsample(d)
		
		# NOTE: d_up should already match frame_text_cond time dimension
		# since both go through same downsample/upsample factors.
		# If mismatched, it indicates an issue with input padding.
		
		# Direct output projection - no normalization (following PortaSpeech)
		out = self.out_proj(d_up)
		recon = ops.transpose(out, (0, 2, 1))
		residual = self.residual_proj(d_up) if self.return_residual else None
		return recon, residual
	
	def get_config(self):
		cfg = super().get_config()
		cfg.update({
			"n_mels": self.n_mels,
			"cond_dim": self.cond_dim,
			"model_channels": self.model_channels,
			"latent_dim": self.latent_dim,
			"num_wavenet_blocks": self.num_wavenet_blocks,
			"decoder_blocks": self.decoder_blocks,
			"wavenet_kernel_size": self.wavenet_kernel_size,
			"down_stages": self.down_stages,
			"dropout": self.dropout_rate,
		})
		return cfg


def create_vae(
	n_mels: int,
	cond_dim: int,
	model_channels: int = 256,
	num_wavenet_blocks: int = 6,
	wavenet_kernel_size: int = 5,
	down_stages: int = 2,
	flow_layers: int = 4,
	flow_hidden: int = 256,
	decoder_channels: Optional[int] = None,
	dropout: float = 0.1,
) -> TextConditionedVAE:
	"""Factory for TextConditionedVAE with sensible defaults."""
	return TextConditionedVAE(
		n_mels=n_mels,
		cond_dim=cond_dim,
		model_channels=model_channels,
		num_wavenet_blocks=num_wavenet_blocks,
		wavenet_kernel_size=wavenet_kernel_size,
		down_stages=down_stages,
		flow_layers=flow_layers,
		flow_hidden=flow_hidden,
		decoder_channels=decoder_channels,
		dropout=dropout,
	)


