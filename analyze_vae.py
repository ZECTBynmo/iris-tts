"""Analyze VAE outputs to diagnose training issues."""

import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax.numpy as jnp
import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from iris.vae import TextConditionedVAE
from iris.encoder import PhonemeEncoder
from iris.text import create_text_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_vocab(vocab_path: str):
    vocab = np.load(vocab_path, allow_pickle=True).item()
    return vocab["phoneme_to_id"], vocab["id_to_phoneme"]


def extract_mel_spectrogram(audio_path: str, n_mels: int = 80, sample_rate: int = 22050):
    """Extract mel-spectrogram from audio."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
        fmin=0,
        fmax=8000,
    )
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return mel, audio


def plot_mel(mel, title, save_path):
    """Plot mel-spectrogram."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=22050, hop_length=256, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved plot: {save_path}")


def main():
    logger.info("=" * 80)
    logger.info("VAE Analysis and Diagnosis")
    logger.info("=" * 80)
    
    # Config
    n_mels = 80
    embed_dim = 256
    down_stages = 2
    sample_rate = 22050
    
    # Load VAE
    logger.info("\n[1/6] Loading VAE model...")
    vae = TextConditionedVAE(
        n_mels=n_mels,
        cond_dim=embed_dim,
        model_channels=embed_dim,
        num_wavenet_blocks=6,
        down_stages=down_stages,
        flow_layers=4,
        flow_hidden=embed_dim,
    )
    
    # Build
    dummy_mel = jnp.ones((1, n_mels, 16), dtype="float32")
    dummy_cond = jnp.ones((1, 16, embed_dim), dtype="float32")
    _ = vae(mels_bt_f=dummy_mel, frame_text_cond=dummy_cond, training=False)
    
    vae_weights = "outputs/vae/checkpoints_vae/vae_core_best.weights.h5"
    if not Path(vae_weights).exists():
        logger.error(f"VAE weights not found at: {vae_weights}")
        return
    
    vae.load_weights(vae_weights)
    logger.info(f"✓ Loaded VAE from: {vae_weights}")
    
    # Load a real mel-spectrogram from LJSpeech
    logger.info("\n[2/6] Loading real audio sample...")
    audio_file = Path("data/LJSpeech-1.1/wavs/LJ001-0001.wav")
    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    real_mel, real_audio = extract_mel_spectrogram(str(audio_file), n_mels=n_mels)
    logger.info(f"Real mel shape: {real_mel.shape}")
    logger.info(f"Real mel stats: min={real_mel.min():.3f}, max={real_mel.max():.3f}, mean={real_mel.mean():.3f}, std={real_mel.std():.3f}")
    
    # Save real mel plot
    output_dir = Path("outputs/vae_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_mel(real_mel, "Real Mel-Spectrogram (Ground Truth)", output_dir / "1_real_mel.png")
    
    # Test 1: VAE Reconstruction (with proper conditioning)
    logger.info("\n[3/6] Testing VAE reconstruction...")
    
    # Prepare mel for VAE (need to pad to multiple of 2^down_stages)
    factor = 2 ** down_stages
    time_steps = real_mel.shape[1]
    padded_time = int(np.ceil(time_steps / factor) * factor)
    
    mel_padded = np.zeros((n_mels, padded_time), dtype=np.float32)
    mel_padded[:, :time_steps] = real_mel
    
    # Convert to VAE format [batch, n_mels, time]
    mel_batch = mel_padded[np.newaxis, ...]
    mel_jax = jnp.array(mel_batch)
    
    # Create dummy text conditioning (all zeros for reconstruction test)
    text_cond = jnp.zeros((1, padded_time, embed_dim), dtype="float32")
    
    # Run through VAE
    logger.info("Running forward pass...")
    try:
        output = vae(mels_bt_f=mel_jax, frame_text_cond=text_cond, training=False)
        
        if isinstance(output, tuple):
            reconstructed_mel_jax = output[0]
            logger.info(f"VAE returned tuple with {len(output)} elements")
        else:
            reconstructed_mel_jax = output
        
        reconstructed_mel = np.array(reconstructed_mel_jax)[0]  # [n_mels, time]
        
        logger.info(f"Reconstructed mel shape: {reconstructed_mel.shape}")
        logger.info(f"Reconstructed mel stats: min={reconstructed_mel.min():.3f}, max={reconstructed_mel.max():.3f}, mean={reconstructed_mel.mean():.3f}, std={reconstructed_mel.std():.3f}")
        
        # Trim to original length
        reconstructed_mel = reconstructed_mel[:, :time_steps]
        
        # Calculate reconstruction error
        mse = np.mean((real_mel - reconstructed_mel) ** 2)
        mae = np.mean(np.abs(real_mel - reconstructed_mel))
        logger.info(f"Reconstruction MSE: {mse:.6f}")
        logger.info(f"Reconstruction MAE: {mae:.6f}")
        
        plot_mel(reconstructed_mel, "VAE Reconstructed Mel", output_dir / "2_reconstructed_mel.png")
        
        # Plot difference
        diff = np.abs(real_mel - reconstructed_mel)
        plot_mel(diff, "Reconstruction Error (Absolute Difference)", output_dir / "3_reconstruction_error.png")
        
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: VAE Generation (unconditional)
    logger.info("\n[4/6] Testing VAE generation (unconditional)...")
    
    try:
        gen_time = 128  # frames
        gen_cond = jnp.zeros((1, gen_time, embed_dim), dtype="float32")
        
        generated_mel_jax, residual = vae.generate(gen_cond)
        generated_mel = np.array(generated_mel_jax)[0]  # [n_mels, time]
        
        logger.info(f"Generated mel shape: {generated_mel.shape}")
        logger.info(f"Generated mel stats: min={generated_mel.min():.3f}, max={generated_mel.max():.3f}, mean={generated_mel.mean():.3f}, std={generated_mel.std():.3f}")
        
        plot_mel(generated_mel, "VAE Generated Mel (Zero Conditioning)", output_dir / "4_generated_uncond.png")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Generation with random conditioning
    logger.info("\n[5/6] Testing VAE generation (random conditioning)...")
    
    try:
        # Random conditioning to see if VAE responds to it
        random_cond = jnp.array(np.random.randn(1, gen_time, embed_dim).astype(np.float32) * 0.1)
        
        generated_mel_jax, residual = vae.generate(random_cond)
        generated_mel_random = np.array(generated_mel_jax)[0]
        
        logger.info(f"Generated mel (random cond) stats: min={generated_mel_random.min():.3f}, max={generated_mel_random.max():.3f}, mean={generated_mel_random.mean():.3f}, std={generated_mel_random.std():.3f}")
        
        plot_mel(generated_mel_random, "VAE Generated Mel (Random Conditioning)", output_dir / "5_generated_random.png")
        
        # Compare with zero conditioning
        diff_cond = np.abs(generated_mel - generated_mel_random)
        logger.info(f"Difference between zero and random conditioning: mean={diff_cond.mean():.6f}, max={diff_cond.max():.6f}")
        
    except Exception as e:
        logger.error(f"Generation with random conditioning failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Check latent space
    logger.info("\n[6/6] Analyzing latent space...")
    
    try:
        # Check what encoder produces
        if hasattr(vae, 'encoder'):
            latent = vae.encoder(mel_jax, text_cond, training=False)
            logger.info(f"Latent shape: {latent.shape}")
            logger.info(f"Latent stats: min={np.array(latent).min():.3f}, max={np.array(latent).max():.3f}, mean={np.array(latent).mean():.3f}, std={np.array(latent).std():.3f}")
            
            # Check if latent is degenerate (all zeros or very small values)
            if np.abs(np.array(latent)).max() < 0.01:
                logger.warning("⚠️ WARNING: Latent values are very small, encoder might not be working properly!")
        
    except Exception as e:
        logger.info(f"Could not analyze latent space: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Plots saved to: {output_dir}/")
    logger.info("\nKey Observations:")
    logger.info(f"1. Real mel range: [{real_mel.min():.3f}, {real_mel.max():.3f}]")
    logger.info(f"2. Reconstructed mel range: [{reconstructed_mel.min():.3f}, {reconstructed_mel.max():.3f}]")
    logger.info(f"3. Reconstruction MSE: {mse:.6f}")
    logger.info(f"4. Reconstruction MAE: {mae:.6f}")
    
    # Diagnostics
    logger.info("\nDiagnostics:")
    if mse > 5.0:
        logger.warning("⚠️ High reconstruction error - VAE may not be trained well")
    elif mse > 1.0:
        logger.warning("⚠️ Moderate reconstruction error - VAE could be better")
    else:
        logger.info("✓ Reconstruction error looks reasonable")
    
    if np.abs(reconstructed_mel.mean()) > 20:
        logger.warning("⚠️ Reconstructed mel has unusual mean value")
    
    if reconstructed_mel.std() < 0.5:
        logger.warning("⚠️ Reconstructed mel has very low variance - might be producing flat outputs")
    
    if generated_mel.std() < 0.5:
        logger.warning("⚠️ Generated mel has very low variance - might be mode collapsed")
    
    logger.info("\nTo investigate further, check the plots in outputs/vae_analysis/")


if __name__ == "__main__":
    main()

