#!/usr/bin/env python3
"""
Quick-start script for aligning LJSpeech dataset using Montreal Forced Aligner.

Usage:
    python align_ljspeech.py
"""

import logging
import subprocess
from pathlib import Path
from iris.alignment import MFAAligner, create_text_files_from_metadata

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_corpus_directory(audio_dir: Path, corpus_dir: Path) -> None:
    """
    Create MFA corpus structure: corpus/SPEAKER/audio.wav + corpus/SPEAKER/audio.lab
    
    Args:
        audio_dir: Source audio directory
        corpus_dir: Output corpus directory
    """
    logger.info("Creating MFA corpus structure...")
    
    corpus_dir.mkdir(parents=True, exist_ok=True)
    speaker_dir = corpus_dir / "LJSpeech"
    speaker_dir.mkdir(exist_ok=True)
    
    # Create symlinks or copy audio files to corpus directory
    audio_files = list(audio_dir.glob("*.wav"))
    logger.info(f"Linking {len(audio_files)} audio files...")
    
    for audio_file in audio_files:
        symlink_path = speaker_dir / audio_file.name
        # Remove if exists
        if symlink_path.exists():
            symlink_path.unlink()
        # Create symlink
        symlink_path.symlink_to(audio_file.absolute())


def download_dictionary() -> Path:
    """
    Download or create CMU dictionary for English.
    Returns path to dictionary file.
    """
    dict_path = Path("data/cmu_dict.txt")
    
    if dict_path.exists():
        logger.info(f"Using existing dictionary: {dict_path}")
        return dict_path
    
    logger.info("Downloading CMU dictionary...")
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use curl to download the CMU dict
    try:
        subprocess.run(
            [
                "curl",
                "-L",
                "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict-0.7b",
                "-o",
                str(dict_path),
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"Dictionary downloaded to: {dict_path}")
    except Exception as e:
        logger.error(f"Failed to download dictionary: {e}")
        logger.info("You can download manually from:")
        logger.info("https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict-0.7b")
        raise
    
    return dict_path


def main():
    # Paths
    data_dir = Path("data/LJSpeech-1.1")
    audio_dir = data_dir / "wavs"
    metadata_file = data_dir / "metadata.csv"
    corpus_dir = Path("data/ljspeech_corpus")
    alignments_dir = Path("data/ljspeech_alignments")
    
    logger.info("=" * 70)
    logger.info("LJSpeech Montreal Forced Aligner Setup")
    logger.info("=" * 70)
    
    # Validation
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return
    
    if not audio_dir.exists() or not list(audio_dir.glob("*.wav")):
        logger.error(f"No .wav files found in: {audio_dir}")
        return
    
    # Step 1: Create corpus structure
    logger.info("\n[1/5] Creating MFA corpus structure...")
    create_corpus_directory(audio_dir, corpus_dir)
    
    # Step 2: Create text files in corpus
    logger.info("\n[2/5] Creating transcription files...")
    create_text_files_from_metadata(str(metadata_file), str(corpus_dir))
    
    # Step 3: Download dictionary
    logger.info("\n[3/5] Setting up pronunciation dictionary...")
    try:
        dict_path = download_dictionary()
    except Exception as e:
        logger.error(f"Dictionary setup failed: {e}")
        return
    
    # Step 4: Initialize and download models
    logger.info("\n[4/5] Downloading MFA models...")
    logger.info("Note: This may take a while on first run (~2-3 GB)")
    
    aligner = MFAAligner(
        model_name="english_mfa_lm",
        acoustic_model="english_us_arpa",
    )
    
    try:
        aligner.download_models()
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        logger.info("You may need to install MFA manually:")
        logger.info("  conda install -c conda-forge montreal-forced-aligner")
        return
    
    # Step 5: Run alignment
    logger.info("\n[5/5] Running forced alignment...")
    logger.info(f"Corpus directory: {corpus_dir}")
    logger.info(f"Dictionary: {dict_path}")
    logger.info(f"Output directory: {alignments_dir}")
    logger.info("This may take 10-30 minutes for 13,100 files...")
    
    try:
        alignments_dir.mkdir(parents=True, exist_ok=True)
        
        # Run MFA align command directly with proper paths
        cmd = [
            "mfa",
            "align",
            str(corpus_dir.absolute()),
            str(dict_path.absolute()),
            "english_us_arpa",
            str(alignments_dir.absolute()),
            "-j", "4",
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Alignment complete!")
        logger.info("=" * 70)
        logger.info(f"\nTextGrid files saved to: {alignments_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Load alignments: alignments = aligner.load_alignments(...)")
        logger.info("2. Use alignments in your TTS training pipeline")
        
    except Exception as e:
        logger.error(f"\nAlignment failed: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("- Ensure conda MFA is installed: conda install -c conda-forge montreal-forced-aligner")
        logger.info("- Check that corpus structure is: data/ljspeech_corpus/SPEAKER/*.wav + *.lab")
        logger.info("- Verify text files are UTF-8 encoded")
        return


if __name__ == "__main__":
    main()

