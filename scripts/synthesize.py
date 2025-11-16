"""Synthesize audio from text using Encoder -> Duration -> VAE -> (PostNet) -> Vocoder."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import logging
from pathlib import Path
import importlib
import numpy as np
import jax.numpy as jnp
import keras

from iris.text import create_text_processor
from iris.encoder import PhonemeEncoder, DurationPredictor
from iris.vae import TextConditionedVAE
from iris.postnet import PostNet
from iris.hifigan_pretrained import get_pretrained_hifigan


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_vocab(vocab_path: str):
    vocab = np.load(vocab_path, allow_pickle=True).item()
    return vocab["phoneme_to_id"], vocab["id_to_phoneme"]


def text_to_ids(text: str, processor, phoneme_to_id: dict) -> np.ndarray:
    phonemes = processor.text_to_phonemes(text).split()
    ids = []
    unk_id = phoneme_to_id.get("<UNK>", 0)
    for p in phonemes:
        ids.append(phoneme_to_id.get(p, unk_id))
    if not ids:
        ids = [unk_id]
    return np.array(ids, dtype=np.int32)


def predict_durations(encoder_out: jnp.ndarray, duration_head: DurationPredictor) -> jnp.ndarray:
    # duration_head outputs log(duration+1); invert and clamp
    pred = duration_head(encoder_out, training=False)  # [B, P, 1]
    frames = jnp.clip(jnp.round(jnp.exp(pred[..., 0]) - 1.0), 1.0, 1e6)
    return frames.astype(jnp.int32)  # [B, P]


def length_regulate_np(encoder_out_np: np.ndarray, durations_np: np.ndarray) -> np.ndarray:
    """Expand phoneme-level encoder outputs to frame-level using numpy.

    Args:
        encoder_out_np: [1, P, E]
        durations_np: [1, P] int frames per phoneme
    Returns:
        frame_cond: [1, T, E]
    """
    enc = encoder_out_np[0]  # [P, E]
    durs = durations_np[0].astype(np.int32)
    durs = np.maximum(durs, 1)
    expanded = np.repeat(enc, durs, axis=0)  # [T, E]
    return expanded[None, ...]


def main():
    parser = argparse.ArgumentParser(description="Synthesize audio from text with VAE pipeline")
    parser.add_argument("--text", type=str, default="Hello world, this is a test.")
    parser.add_argument("--output_wav", type=str, default="outputs/sample.wav")
    parser.add_argument("--vocab_path", type=str, default="outputs/vae/cache/phoneme_vocab.npy")
    parser.add_argument("--encoder_weights", type=str, default="outputs/encoder/checkpoints/encoder_best.weights.h5")
    parser.add_argument("--duration_weights", type=str, default="outputs/encoder/checkpoints/duration_best.weights.h5")
    parser.add_argument("--vae_core_weights", type=str, default="outputs/vae/checkpoints_vae/vae_core_best.weights.h5")
    parser.add_argument("--postnet_weights", type=str, default="outputs/vae/checkpoints_postnet/postnet_best.weights.h5")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--down_stages", type=int, default=2)
    parser.add_argument("--target_len", type=int, default=1024)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--use_griffin_lim", action="store_true", help="Use Griffin-Lim instead of HiFiGAN")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    # RNG seed for VAE
    keras.utils.set_random_seed(args.seed)

    # Load vocab and text processor
    phoneme_to_id, id_to_phoneme = load_vocab(args.vocab_path)
    vocab_size = len(phoneme_to_id)
    processor = create_text_processor(use_g2p=True, use_nemo=True)
    phoneme_ids_np = text_to_ids(args.text, processor, phoneme_to_id)[None, :]  # [1, P]
    phoneme_ids = jnp.array(phoneme_ids_np)

    # Encoder and duration
    encoder = PhonemeEncoder(vocab_size=vocab_size, embed_dim=args.embed_dim, num_blocks=4, num_heads=4, dropout=0.1)
    _ = encoder(jnp.ones((1, 8), dtype="int32"), training=False)
    logger.info(f"Loading encoder weights: {Path(args.encoder_weights).resolve()}")
    encoder.load_weights(args.encoder_weights)
    duration_head = DurationPredictor(hidden_dim=args.embed_dim, dropout=0.1)
    # Build duration head with dummy
    _ = duration_head(jnp.ones((1, 8, args.embed_dim), dtype="float32"), training=False)
    if Path(args.duration_weights).exists():
        logger.info(f"Loading duration predictor weights: {Path(args.duration_weights).resolve()}")
        duration_head.load_weights(args.duration_weights)
    else:
        logger.warning("Duration weights not found; durations may be poor.")

    # Encode and predict durations
    enc_out = encoder(phoneme_ids, training=False)  # [1, P, E]
    pred_frames = predict_durations(enc_out, duration_head)  # [1, P]

    # Use NumPy-based regulator to avoid JAX tracing issues at inference
    enc_out_np = np.array(enc_out)
    pred_frames_np = np.array(pred_frames)
    frame_cond_np = length_regulate_np(enc_out_np, pred_frames_np)

    # Fix length to multiple of factor
    factor = 2 ** args.down_stages
    T = int(np.ceil(frame_cond_np.shape[1] / factor) * factor)
    if frame_cond_np.shape[1] < T:
        pad = np.zeros((1, T - frame_cond_np.shape[1], args.embed_dim), dtype=frame_cond_np.dtype)
        frame_cond_np = np.concatenate([frame_cond_np, pad], axis=1)
    frame_cond = jnp.array(frame_cond_np)

    # VAE (matching PortaSpeech architecture)
    vae = TextConditionedVAE(
        n_mels=args.n_mels,
        cond_dim=args.embed_dim,
        model_channels=192,
        latent_dim=16,
        num_wavenet_blocks=8,
        decoder_blocks=4,
        down_stages=args.down_stages,
        flow_layers=4,
        flow_hidden=64,
    )
    _ = vae(
        mels_bt_f=jnp.ones((1, args.n_mels, 16), dtype="float32"),
        frame_text_cond=jnp.ones((1, 16, args.embed_dim), dtype="float32"),
        training=False,
    )
    logger.info(f"Loading VAE core: {Path(args.vae_core_weights).resolve()}")
    vae.load_weights(args.vae_core_weights)

    # Generate mel
    mel_bt_f, residual = vae.generate(frame_cond)  # [1, n_mels, T]

    # Optional PostNet
    postnet_path = Path(args.postnet_weights)
    if postnet_path.exists() and postnet_path.stat().st_size > 1024:  # Check file exists and not empty
        logger.info(f"Loading PostNet: {postnet_path.resolve()}")
        # Create PostNet with same architecture as current training
        postnet = PostNet(
            n_mels=args.n_mels,
            num_layers=3,      # Current training
            channels=256,      # Current training
            kernel_size=5,
            dropout=0.3,       # Current training
        )
        # Build PostNet with realistic size and training=True to init all BatchNorm stats
        mel_shape = mel_bt_f.shape
        dummy_mel = jnp.ones((1, args.n_mels, mel_shape[2]), dtype="float32")
        _ = postnet(dummy_mel, training=True)  # training=True initializes BatchNorm properly
        # Load weights
        postnet.load_weights(str(postnet_path))
        # Apply refinement
        mel_bt_f = postnet(mel_bt_f, training=False)
        logger.info("✓ PostNet refinement applied")
    else:
        logger.info("No valid PostNet weights found, using VAE output directly")

    mel_bt_f_np = np.array(mel_bt_f)

    # Vocoder
    if args.use_griffin_lim:
        logger.info("Using Griffin-Lim vocoder...")
        # Griffin-Lim expects linear magnitude mel
        m_log = mel_bt_f_np[0]  # [n_mels, T]
        
        # Clip extreme values before exp to prevent overflow
        m_log_clipped = np.clip(m_log, -11.513, 2.0)  # Reasonable range for log mels
        m_lin = np.exp(m_log_clipped).astype(float)
        
        import librosa
        import librosa.feature
        
        # mel_to_stft needs power=1.0 since VAE was trained with magnitude
        S = librosa.feature.inverse.mel_to_stft(
            m_lin, 
            sr=args.sample_rate, 
            n_fft=1024,
            power=1.0  # Match training (magnitude, not power)
        )
        audio = librosa.griffinlim(S, n_iter=60, hop_length=args.hop_length, win_length=1024)
        audio = audio.astype(np.float32)
    else:
        logger.info("Using HiFiGAN vocoder...")
        vocoder = get_pretrained_hifigan()
        audio = vocoder(mel_bt_f_np)
        audio = np.asarray(audio, dtype=np.float32)
        
        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.squeeze()
    
    logger.info(f"Generated audio: {audio.shape}, duration={len(audio)/args.sample_rate:.2f}s")

    # Save WAV
    out_path = Path(args.output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(out_path), audio, args.sample_rate)
        logger.info(f"Wrote {out_path}")
    except Exception as e:
        np.save(str(out_path.with_suffix(".npy")), audio)
        logger.warning(f"soundfile not available or save failed; wrote numpy array instead: {out_path.with_suffix('.npy')} ({e})")


if __name__ == "__main__":
    main()


