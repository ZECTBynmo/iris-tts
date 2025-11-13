"""Forced alignment utilities using Montreal Forced Aligner."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List
import subprocess
import logging

logger = logging.getLogger(__name__)


class MFAAligner:
    """Montreal Forced Aligner wrapper for phoneme-level alignment."""
    
    def __init__(
        self,
        model_name: str = "english_mfa_lm",
        acoustic_model: str = "english_us_arpa",
    ):
        """
        Initialize the MFA aligner.
        
        Args:
            model_name: Name of the language model to use (default: english_mfa_lm)
            acoustic_model: Name of the acoustic model to use (default: english_us_arpa)
        """
        self.model_name = model_name
        self.acoustic_model = acoustic_model
        self.language = "english"
        
    def prepare_data(
        self,
        audio_dir: str,
        text_dir: str,
        output_dir: str,
    ) -> None:
        """
        Prepare data for alignment (validate structure).
        
        Args:
            audio_dir: Directory containing .wav files
            text_dir: Directory containing .lab files (one line of text per file)
            output_dir: Output directory for alignments
        """
        audio_dir = Path(audio_dir)
        text_dir = Path(text_dir)
        output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate that audio and text files match
        audio_files = set(f.stem for f in audio_dir.glob("*.wav"))
        text_files = set(f.stem for f in text_dir.glob("*.lab"))
        
        missing_text = audio_files - text_files
        missing_audio = text_files - audio_files
        
        if missing_text:
            logger.warning(f"Missing text files for: {missing_text}")
        if missing_audio:
            logger.warning(f"Missing audio files for: {missing_audio}")
        
        logger.info(f"Found {len(audio_files & text_files)} matching audio-text pairs")
    
    def download_models(self) -> None:
        """Download required MFA models."""
        logger.info("Downloading MFA models...")
        try:
            subprocess.run(
                ["mfa", "model", "download", "acoustic", self.acoustic_model],
                check=True,
            )
            subprocess.run(
                ["mfa", "model", "download", "language_model", self.model_name],
                check=True,
            )
            logger.info("Models downloaded successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download models: {e}")
            raise
    
    def align(
        self,
        audio_dir: str,
        text_dir: str,
        output_dir: str,
        num_jobs: int = 4,
    ) -> None:
        """
        Perform forced alignment on the dataset.
        
        Args:
            audio_dir: Directory containing .wav files
            text_dir: Directory containing .lab files with transcriptions
            output_dir: Output directory for alignments
            num_jobs: Number of parallel jobs
        """
        logger.info(f"Starting alignment with {num_jobs} jobs...")
        
        # Convert to absolute paths
        audio_dir = Path(audio_dir).absolute()
        text_dir = Path(text_dir).absolute()
        output_dir = Path(output_dir).absolute()
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            cmd = [
                "mfa",
                "align",
                str(audio_dir),
                str(text_dir),
                self.acoustic_model,
                str(output_dir),
                "-j", str(num_jobs),
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"Alignment complete! Results in {output_dir}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Alignment failed: {e}")
            raise
    
    def load_alignments(self, output_dir: str) -> Dict[str, List[Dict]]:
        """
        Load TextGrid alignments from MFA output.
        
        Args:
            output_dir: Directory containing .TextGrid files
            
        Returns:
            Dictionary mapping filenames to phoneme alignment data
        """
        try:
            from textgrid import TextGrid
        except ImportError:
            logger.error(
                "textgrid package required. Install with: pip install textgrid"
            )
            raise
        
        output_dir = Path(output_dir)
        alignments = {}
        
        for textgrid_file in output_dir.glob("*.TextGrid"):
            try:
                tg = TextGrid.fromFile(str(textgrid_file))
                name = textgrid_file.stem
                
                # Extract phone-level alignments from the tier
                phones = []
                if len(tg.tiers) > 0:
                    phone_tier = tg.tiers[0]
                    for interval in phone_tier:
                        if interval.mark:  # Skip silence
                            phones.append({
                                "phone": interval.mark,
                                "start": interval.minTime,
                                "end": interval.maxTime,
                                "duration": interval.maxTime - interval.minTime,
                            })
                
                alignments[name] = phones
                logger.debug(f"Loaded {len(phones)} phones for {name}")
                
            except Exception as e:
                logger.error(f"Failed to load {textgrid_file}: {e}")
        
        logger.info(f"Loaded {len(alignments)} alignment files")
        return alignments


def create_text_files_from_metadata(
    metadata_file: str,
    output_dir: str,
) -> None:
    """
    Create .lab text files from LJSpeech metadata.csv.
    MFA 3.x expects corpus directory structure with speaker subdirectories.
    
    Args:
        metadata_file: Path to metadata.csv
        output_dir: Output corpus directory (will create speaker subdirs)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create speaker directory (LJSpeech is single speaker)
    speaker_dir = output_dir / "LJSpeech"
    speaker_dir.mkdir(exist_ok=True)
    
    with open(metadata_file, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                filename = parts[0]
                # Use the normalized text (column 3)
                text = parts[2].strip()
                
                lab_file = speaker_dir / f"{filename}.lab"
                with open(lab_file, "w") as lab_f:
                    lab_f.write(text)
    
    logger.info(f"Created {len(list(speaker_dir.glob('*.lab')))} text files in {speaker_dir}")

