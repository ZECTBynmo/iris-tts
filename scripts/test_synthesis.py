"""End-to-end synthesis test on real LJSpeech samples.

This script runs the full stack:
  Text/phonemes (from MFA alignments via LJSpeechDurationDataset) ->
  PhonemeEncoder ->
  DurationPredictor ->
  VAE (TextConditionedVAE) ->
  (optional) PostNet ->
  Vocoder

It then compares the generated mel-spectrogram against the ground-truth mel
computed from the original audio for the same sample, and reports simple
error metrics (MSE / MAE).
"""

import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import logging
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import keras

from iris.datasets import LJSpeechDurationDataset
from iris.encoder import PhonemeEncoder, DurationPredictor
from iris.vae import TextConditionedVAE
from iris.postnet import PostNet
from iris.hifigan_pretrained import get_pretrained_hifigan
from iris.data import load_audio, compute_mel_spectrogram


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def length_regulate_np(encoder_out_np: np.ndarray, durations_np: np.ndarray) -> np.ndarray:
    """Expand phoneme-level encoder outputs to frame-level using numpy.

    Args:
        encoder_out_np: [B, P, E]
        durations_np: [B, P] int frames per phoneme
    Returns:
        frame_cond: [B, T, E]
    """
    batch_size, _, embed_dim = encoder_out_np.shape
    outs = []
    for b in range(batch_size):
        enc = encoder_out_np[b]  # [P, E]
        durs = durations_np[b].astype(np.int32)
        durs = np.maximum(durs, 1)
        expanded = np.repeat(enc, durs, axis=0)  # [T, E]
        outs.append(expanded)
    max_t = max(o.shape[0] for o in outs)
    frame_cond = np.zeros((batch_size, max_t, embed_dim), dtype=encoder_out_np.dtype)
    for b, o in enumerate(outs):
        frame_cond[b, : o.shape[0]] = o
    return frame_cond


def predict_durations(enc_out: jnp.ndarray, duration_head: DurationPredictor) -> jnp.ndarray:
    """Invert duration_head outputs (log(duration+1)) to frame counts."""
    pred = duration_head(enc_out, training=False)  # [B, P, 1]
    frames = jnp.clip(jnp.round(jnp.exp(pred[..., 0]) - 1.0), 1.0, 1e6)
    return frames.astype(jnp.int32)  # [B, P]


def build_vae_from_config(config_path: Path, weights_path: Path) -> TextConditionedVAE:
    """Construct VAE using saved training config and load weights."""
    import json

    with open(config_path, "r") as f:
        cfg = json.load(f)

    vae = TextConditionedVAE(
        n_mels=cfg["n_mels"],
        cond_dim=cfg["embed_dim"],
        model_channels=cfg["model_channels"],
        latent_dim=cfg.get("latent_dim", 16),
        num_wavenet_blocks=cfg.get("num_blocks", 8),
        decoder_blocks=cfg.get("decoder_blocks", 4),
        down_stages=cfg.get("down_stages", 2),
        flow_layers=cfg.get("flow_layers", 4),
        flow_hidden=cfg.get("flow_hidden", 64),
    )

    # Build with dummy input so weights shapes are initialized
    _ = vae(
        mels_bt_f=jnp.ones((1, cfg["n_mels"], 16), dtype="float32"),
        frame_text_cond=jnp.ones((1, 16, cfg["embed_dim"]), dtype="float32"),
        training=False,
    )

    logger.info(f"Loading VAE core: {weights_path.resolve()}")
    vae.load_weights(str(weights_path))
    return vae


def main():
    parser = argparse.ArgumentParser(description="Test end-to-end synthesis on LJSpeech samples")
    parser.add_argument("--ljspeech_dir", type=str, default="data/LJSpeech-1.1")
    parser.add_argument("--alignments_dir", type=str, default="data/ljspeech_alignments/LJSpeech")
    parser.add_argument("--encoder_cache_dir", type=str, default="outputs/encoder/cache")
    parser.add_argument("--encoder_weights", type=str, default="outputs/encoder/checkpoints/encoder_best.weights.h5")
    parser.add_argument("--duration_weights", type=str, default="outputs/encoder/checkpoints/duration_best.weights.h5")
    parser.add_argument("--vae_config", type=str, default="outputs/vae/config_vae.json")
    parser.add_argument("--vae_core_weights", type=str, default="outputs/vae/checkpoints_vae/vae_core_best.weights.h5")
    parser.add_argument("--postnet_weights", type=str, default="outputs/vae/checkpoints_postnet/postnet_best.weights.h5")
    parser.add_argument("--sample_index", type=int, default=0, help="Index into LJSpeechDurationDataset val split")
    parser.add_argument("--use_griffin_lim", action="store_true", help="Use Griffin-Lim instead of HiFiGAN")
    parser.add_argument("--output_prefix", type=str, default="outputs/test_synthesis")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.seed)

    ljspeech_dir = Path(args.ljspeech_dir)
    alignments_dir = Path(args.alignments_dir)

    # -------------------------------------------
    # Dataset & reference audio/mels
    # -------------------------------------------
    logger.info("Loading LJSpeechDurationDataset (val split)...")
    duration_dataset = LJSpeechDurationDataset(
        ljspeech_dir=str(ljspeech_dir),
        alignments_dir=str(alignments_dir),
        split="val",
        val_split=0.05,
        cache_dir=args.encoder_cache_dir,
    )

    if args.sample_index < 0 or args.sample_index >= len(duration_dataset):
        raise IndexError(f"sample_index {args.sample_index} out of range (0..{len(duration_dataset)-1})")

    sample = duration_dataset[args.sample_index]
    phoneme_ids_np = sample["phoneme_ids"][None, :]  # [1, P]
    durations_gt_np = sample["durations"][None, :]   # [1, P]
    text = sample["text"]
    file_id = sample["file_id"]

    logger.info(f"Selected sample index {args.sample_index} | id={file_id} | text='{text}'")

    # Ground-truth audio & mel
    audio_path = ljspeech_dir / "wavs" / f"{file_id}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Loading reference audio: {audio_path}")
    audio = load_audio(str(audio_path), sample_rate=22050)
    mel_ref = compute_mel_spectrogram(
        audio,
        sample_rate=22050,
        hop_length=256,
        n_mels=80,
    )  # [n_mels, T_ref]
    logger.info(
        f"Reference mel shape: {mel_ref.shape} | "
        f"range=[{mel_ref.min():.3f}, {mel_ref.max():.3f}], "
        f"mean={mel_ref.mean():.3f}, std={mel_ref.std():.3f}"
    )

    # -------------------------------------------
    # Encoder and DurationPredictor
    # -------------------------------------------
    vocab_size = duration_dataset.get_vocab_size()
    embed_dim = 256

    logger.info("Building phoneme encoder...")
    encoder = PhonemeEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_blocks=4,
        num_heads=4,
        dropout=0.1,
    )
    _ = encoder(jnp.ones((1, 8), dtype="int32"), training=False)

    encoder_weights = Path(args.encoder_weights)
    if encoder_weights.exists():
        logger.info(f"Loading encoder weights: {encoder_weights.resolve()}")
        encoder.load_weights(str(encoder_weights))
    else:
        logger.warning(f"Encoder weights not found at {encoder_weights}, using random initialization.")

    logger.info("Building duration predictor...")
    duration_head = DurationPredictor(hidden_dim=embed_dim, dropout=0.1)
    _ = duration_head(jnp.ones((1, 8, embed_dim), dtype="float32"), training=False)

    duration_weights = Path(args.duration_weights)
    if duration_weights.exists():
        logger.info(f"Loading duration predictor weights: {duration_weights.resolve()}")
        duration_head.load_weights(str(duration_weights))
    else:
        logger.warning(f"Duration weights not found at {duration_weights}, using random initialization.")

    phoneme_ids = jnp.array(phoneme_ids_np)

    # Encode and predict durations
    enc_out = encoder(phoneme_ids, training=False)        # [1, P, E]
    pred_durations = predict_durations(enc_out, duration_head)  # [1, P]

    enc_out_np = np.array(enc_out, dtype=np.float32)
    pred_durations_np = np.array(pred_durations, dtype=np.int32)

    logger.info(
        f"Predicted durations stats | "
        f"min={pred_durations_np.min()}, max={pred_durations_np.max()}, "
        f"mean={pred_durations_np.mean():.2f}, sum={pred_durations_np.sum()}"
    )

    # Length regulate to frame-level conditioning
    frame_cond_np = length_regulate_np(enc_out_np, pred_durations_np)  # [1, T_cond, E]

    # -------------------------------------------
    # VAE (and PostNet)
    # -------------------------------------------
    vae_config_path = Path(args.vae_config)
    vae_weights_path = Path(args.vae_core_weights)
    vae = build_vae_from_config(vae_config_path, vae_weights_path)

    # Pad conditioning length to multiple of VAE downsample factor
    down_stages = vae.down_stages
    factor = 2 ** down_stages
    T_cond = frame_cond_np.shape[1]
    T_target = int(np.ceil(T_cond / factor) * factor)
    if T_cond < T_target:
        pad = np.zeros(
            (frame_cond_np.shape[0], T_target - T_cond, frame_cond_np.shape[2]),
            dtype=frame_cond_np.dtype,
        )
        frame_cond_np = np.concatenate([frame_cond_np, pad], axis=1)

    frame_cond = jnp.array(frame_cond_np)

    # Generate mel from VAE
    mel_gen_bt_f, residual = vae.generate(frame_cond)  # [1, n_mels, T_gen]
    mel_gen = np.array(mel_gen_bt_f[0], dtype=np.float32)  # [n_mels, T_gen]

    logger.info(
        f"Generated VAE mel (pre-PostNet) shape: {mel_gen.shape} | "
        f"range=[{mel_gen.min():.3f}, {mel_gen.max():.3f}], "
        f"mean={mel_gen.mean():.3f}, std={mel_gen.std():.3f}"
    )

    # Optional PostNet
    postnet_path = Path(args.postnet_weights)
    if postnet_path.exists() and postnet_path.stat().st_size > 1024:
        logger.info(f"Loading PostNet: {postnet_path.resolve()}")
        postnet = PostNet(
            n_mels=mel_gen_bt_f.shape[1],
            num_layers=3,
            channels=256,
            kernel_size=5,
            dropout=0.3,
        )
        # Build PostNet with realistic size
        _ = postnet(jnp.ones_like(mel_gen_bt_f), training=True)
        postnet.load_weights(str(postnet_path))
        mel_gen_bt_f = postnet(mel_gen_bt_f, training=False)
        mel_gen = np.array(mel_gen_bt_f[0], dtype=np.float32)
        logger.info(
            f"Generated mel (PostNet) shape: {mel_gen.shape} | "
            f"range=[{mel_gen.min():.3f}, {mel_gen.max():.3f}], "
            f"mean={mel_gen.mean():.3f}, std={mel_gen.std():.3f}"
        )
    else:
        logger.info("No valid PostNet weights found, using VAE output directly.")

    # -------------------------------------------
    # Compare generated mel vs reference mel
    # -------------------------------------------
    T_ref = mel_ref.shape[1]
    T_gen = mel_gen.shape[1]
    T_eval = min(T_ref, T_gen)

    mel_ref_eval = mel_ref[:, :T_eval]
    mel_gen_eval = mel_gen[:, :T_eval]

    mse = float(np.mean((mel_ref_eval - mel_gen_eval) ** 2))
    mae = float(np.mean(np.abs(mel_ref_eval - mel_gen_eval)))

    logger.info(
        f"[EVAL] Mel comparison over {T_eval} frames | "
        f"MSE={mse:.6f}, MAE={mae:.6f}"
    )

    # -------------------------------------------
    # Vocoder: synthesize audio from generated mel
    # -------------------------------------------
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.use_griffin_lim:
        logger.info("Using Griffin-Lim vocoder for generated mel...")
        m_log = mel_gen  # [n_mels, T]
        # Clip to reasonable range before exp
        m_log_clipped = np.clip(m_log, -11.513, 2.0)
        m_lin = np.exp(m_log_clipped).astype(float)

        import librosa
        import librosa.feature

        S = librosa.feature.inverse.mel_to_stft(
            m_lin,
            sr=22050,
            n_fft=1024,
            power=1.0,
        )
        audio_gen = librosa.griffinlim(S, n_iter=60, hop_length=256, win_length=1024)
        audio_gen = audio_gen.astype(np.float32)
    else:
        logger.info("Using HiFiGAN vocoder for generated mel...")
        vocoder = get_pretrained_hifigan()
        audio_gen = vocoder(mel_gen[np.newaxis, ...])
        audio_gen = np.asarray(audio_gen, dtype=np.float32)
        if audio_gen.ndim > 1:
            audio_gen = audio_gen.squeeze()

    logger.info(f"Generated audio shape: {audio_gen.shape}, duration={len(audio_gen)/22050:.2f}s")

    # Save generated and reference audio for listening
    try:
        import soundfile as sf

        gen_path = out_prefix.with_suffix(".wav")
        ref_path = out_prefix.with_name(out_prefix.name + "_ref").with_suffix(".wav")

        sf.write(str(gen_path), audio_gen, 22050)
        sf.write(str(ref_path), audio, 22050)
        logger.info(f"Wrote generated audio: {gen_path}")
        logger.info(f"Wrote reference audio: {ref_path}")
    except Exception as e:
        logger.warning(f"Could not write WAV files (soundfile missing?): {e}")


if __name__ == "__main__":
    main()


