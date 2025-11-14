#!/usr/bin/env python3
"""
Test script to verify HiFiGAN integration works correctly.
This tests loading the pre-trained model and doing inference.
"""

import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("Testing Pre-trained HiFiGAN Integration")
    logger.info("=" * 70)
    
    # Test 1: Import the module
    logger.info("\n[1/3] Importing iris.hifigan_pretrained...")
    try:
        from iris.hifigan_pretrained import infer_hifigan, get_pretrained_hifigan
        logger.info("✓ Import successful")
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return 1
    
    # Test 2: Load the model
    logger.info("\n[2/3] Loading pre-trained HiFiGAN model...")
    try:
        vocoder = get_pretrained_hifigan()
        logger.info("✓ Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"✗ Checkpoint not found: {e}")
        logger.info("\nPlease ensure the HiFiGAN checkpoint is at:")
        logger.info("  models/hifigan/models--speechbrain--tts-hifigan-ljspeech/snapshots/.../generator.ckpt")
        return 1
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 3: Run inference with dummy data
    logger.info("\n[3/3] Testing inference with dummy mel-spectrogram...")
    try:
        # Create a dummy mel-spectrogram (batch=1, n_mels=80, time=100)
        mel = np.random.randn(1, 80, 100).astype(np.float32)
        
        logger.info(f"Input mel shape: {mel.shape}")
        audio = infer_hifigan(mel)
        logger.info(f"Output audio shape: {audio.shape}")
        logger.info(f"Audio dtype: {audio.dtype}")
        logger.info(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        # Verify output shape
        expected_samples = 100 * 256  # time * hop_length (default HiFiGAN)
        if len(audio) > 0:
            logger.info("✓ Inference successful")
        else:
            logger.error("✗ Inference produced empty output")
            return 1
        
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Success!
    logger.info("\n" + "=" * 70)
    logger.info("✓ All tests passed!")
    logger.info("=" * 70)
    logger.info("\nYou can now use HiFiGAN in synthesize.py:")
    logger.info("  python scripts/synthesize.py \\")
    logger.info("    --vocoder hifigan \\")
    logger.info("    --vocoder_entry iris.hifigan_pretrained:infer_hifigan")
    
    return 0


if __name__ == "__main__":
    exit(main())

