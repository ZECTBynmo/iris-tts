"""Tacotron2-style PostNet for mel refinement."""

import keras
from keras import layers, ops
from typing import Optional


class PostNet(keras.Model):
	"""
	Tacotron2-style PostNet:
	- 5 Conv1D layers over time with n_mels channels
	- First 4: Conv + BatchNorm + Tanh + Dropout
	- Last: Conv + BatchNorm (no activation)
	Outputs a residual added to input mel.
	"""
	def __init__(
		self,
		n_mels: int,
		num_layers: int = 4,
		channels: int = 256,
		kernel_size: int = 5,
		dropout: float = 0.5,
		name: Optional[str] = None,
	):
		super().__init__(name=name)
		assert num_layers >= 2, "PostNet needs at least 2 layers"
		self.n_mels = n_mels
		self.num_layers = num_layers
		self.channels = channels
		self.kernel_size = kernel_size
		self.dropout_rate = dropout
		
		self.convs = []
		self.bns = []
		self.dropouts = []
		
		# First L-1 layers
		for i in range(num_layers - 1):
			in_ch = n_mels if i == 0 else channels
			self.convs.append(layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same"))
			self.bns.append(layers.BatchNormalization())
			self.dropouts.append(layers.Dropout(dropout))
		
		# Last layer projects back to n_mels
		self.conv_out = layers.Conv1D(filters=n_mels, kernel_size=kernel_size, padding="same")
		self.bn_out = layers.BatchNormalization()
	
	def call(self, mels_bt_f, training=False):
		"""
		Args:
			mels_bt_f: [B, n_mels, T]
		Returns:
			refined: [B, n_mels, T] (input + residual)
		"""
		# Conv1D expects [B, T, C], transpose
		x = ops.transpose(mels_bt_f, (0, 2, 1))  # [B, T, n_mels]
		h = x
		for conv, bn, do in zip(self.convs, self.bns, self.dropouts):
			h = conv(h)
			h = bn(h, training=training)
			h = ops.tanh(h)
			h = do(h, training=training)
		res = self.conv_out(h)
		res = self.bn_out(res, training=training)
		# Back to [B, n_mels, T]
		res_bt_f = ops.transpose(res, (0, 2, 1))
		return mels_bt_f + res_bt_f
	
	def get_config(self):
		cfg = super().get_config()
		cfg.update({
			"n_mels": self.n_mels,
			"num_layers": self.num_layers,
			"channels": self.channels,
			"kernel_size": self.kernel_size,
			"dropout": self.dropout_rate,
		})
		return cfg


