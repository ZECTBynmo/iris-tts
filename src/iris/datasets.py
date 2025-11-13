"""Dataset download and preparation utilities."""

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TqdmUpTo(tqdm):
    """Progress bar for file downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path, desc: Optional[str] = None) -> Path:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
        
    Returns:
        Path to downloaded file
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if desc is None:
        desc = dest_path.name
    
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)
    
    return dest_path


def extract_tar(tar_path: Path, extract_to: Path, remove_tar: bool = False) -> Path:
    """
    Extract a tar.gz file.
    
    Args:
        tar_path: Path to tar.gz file
        extract_to: Directory to extract to
        remove_tar: Whether to remove tar file after extraction
        
    Returns:
        Path to extracted directory
    """
    tar_path = Path(tar_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {tar_path.name} to {extract_to}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    
    if remove_tar:
        tar_path.unlink()
    
    # Find the extracted directory (usually has the same name as tar without extension)
    extracted_name = tar_path.stem.replace('.tar', '')
    extracted_dir = extract_to / extracted_name
    
    if extracted_dir.exists():
        return extracted_dir
    else:
        # If exact name doesn't match, return the extract_to directory
        return extract_to


def download_ljspeech(
    data_dir: Optional[Path] = None,
    extract: bool = True,
    remove_tar: bool = False,
) -> Path:
    """
    Download LJSpeech dataset.
    
    LJSpeech is a public domain speech dataset consisting of 13,100 short audio clips
    of a single speaker reading passages from 7 non-fiction books.
    
    Args:
        data_dir: Directory to download dataset to. Defaults to ./data
        extract: Whether to extract the tar.gz file
        remove_tar: Whether to remove tar file after extraction (only if extract=True)
        
    Returns:
        Path to LJSpeech dataset directory
    """
    if data_dir is None:
        data_dir = Path("./data")
    else:
        data_dir = Path(data_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # LJSpeech download URL (using a common mirror)
    ljspeech_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    
    tar_path = data_dir / "LJSpeech-1.1.tar.bz2"
    
    # Check if already extracted
    ljspeech_dir = data_dir / "LJSpeech-1.1"
    if ljspeech_dir.exists() and (ljspeech_dir / "wavs").exists():
        print(f"LJSpeech dataset already exists at {ljspeech_dir}")
        return ljspeech_dir
    
    # Download if not exists
    if not tar_path.exists():
        print(f"Downloading LJSpeech dataset from {ljspeech_url}...")
        download_file(ljspeech_url, tar_path, desc="LJSpeech")
    else:
        print(f"Tar file already exists at {tar_path}")
    
    # Extract if requested
    if extract:
        # Handle .tar.bz2 files (LJSpeech uses bzip2 compression)
        if tar_path.suffix == '.bz2' or '.bz2' in tar_path.suffixes:
            print(f"Extracting {tar_path.name} to {data_dir}...")
            with tarfile.open(tar_path, 'r:bz2') as tar:
                tar.extractall(data_dir)
            
            if remove_tar:
                tar_path.unlink()
        elif tar_path.suffix == '.gz' or '.gz' in tar_path.suffixes:
            # Fallback to regular extract_tar for .tar.gz
            extract_tar(tar_path, data_dir, remove_tar=remove_tar)
        else:
            # Try auto-detection
            print(f"Extracting {tar_path.name} to {data_dir}...")
            with tarfile.open(tar_path, 'r:*') as tar:
                tar.extractall(data_dir)
            
            if remove_tar:
                tar_path.unlink()
        
        return ljspeech_dir
    
    return tar_path


def get_ljspeech_path(data_dir: Optional[Path] = None) -> Path:
    """
    Get path to LJSpeech dataset, downloading if necessary.
    
    Args:
        data_dir: Directory containing dataset. Defaults to ./data
        
    Returns:
        Path to LJSpeech dataset directory
    """
    if data_dir is None:
        data_dir = Path("./data")
    else:
        data_dir = Path(data_dir)
    
    ljspeech_dir = data_dir / "LJSpeech-1.1"
    
    if not ljspeech_dir.exists():
        print("LJSpeech dataset not found. Downloading...")
        return download_ljspeech(data_dir=data_dir, extract=True)
    
    return ljspeech_dir


class LJSpeechDurationDataset:
    """
    PyTorch-style dataset for encoder duration prediction training.
    
    Loads MFA alignments and creates training examples of:
    - Phoneme IDs (from text)
    - Duration targets (from MFA alignments in frames)
    """
    
    def __init__(
        self,
        ljspeech_dir: str,
        alignments_dir: str,
        split: str = 'train',
        val_split: float = 0.05,
        sample_rate: int = 22050,
        hop_length: int = 256,
        max_phoneme_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize LJSpeech duration dataset.
        
        Args:
            ljspeech_dir: Path to LJSpeech-1.1 directory
            alignments_dir: Path to MFA alignments directory (e.g., data/ljspeech_alignments/LJSpeech)
            split: 'train' or 'val'
            val_split: Fraction of data to use for validation
            sample_rate: Audio sample rate (default: 22050 Hz)
            hop_length: STFT hop length for frame conversion (default: 256)
            max_phoneme_length: Maximum phoneme sequence length (for filtering)
            cache_dir: Optional directory to cache processed data
        """
        from iris.alignment import MFAAligner
        
        self.ljspeech_dir = Path(ljspeech_dir)
        self.alignments_dir = Path(alignments_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_phoneme_length = max_phoneme_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load metadata
        logger.info("Loading LJSpeech metadata...")
        self.metadata = self._load_metadata()
        
        # Load MFA alignments
        logger.info(f"Loading MFA alignments from {alignments_dir}...")
        aligner = MFAAligner()
        self.alignments = aligner.load_alignments(str(self.alignments_dir))
        logger.info(f"Loaded {len(self.alignments)} alignments")
        
        # Filter to only samples with both metadata and alignments
        valid_ids = set(self.metadata.keys()) & set(self.alignments.keys())
        logger.info(f"Found {len(valid_ids)} samples with both metadata and alignments")
        
        # Split into train/val
        all_ids = sorted(valid_ids)
        val_size = int(len(all_ids) * val_split)
        
        if split == 'val':
            self.sample_ids = all_ids[:val_size]
        else:  # train
            self.sample_ids = all_ids[val_size:]
        
        logger.info(f"{split} split: {len(self.sample_ids)} samples")
        
        # Build phoneme vocabulary from MFA alignments
        if split == 'train':
            logger.info("Building phoneme vocabulary from MFA alignments...")
            
            # Collect all unique phonemes from alignments
            all_phonemes = set()
            for sid in self.sample_ids:
                for phone_info in self.alignments[sid]:
                    all_phonemes.add(phone_info['phone'])
            
            # Sort phonemes for consistent ordering
            phonemes_sorted = sorted(all_phonemes)
            
            # Build mappings with special tokens
            self.phoneme_to_id = {}
            self.id_to_phoneme = {}
            
            # Add special tokens
            special_tokens = ['<PAD>', '<UNK>']
            for i, token in enumerate(special_tokens):
                self.phoneme_to_id[token] = i
                self.id_to_phoneme[i] = token
            
            # Add phonemes
            for phoneme in phonemes_sorted:
                idx = len(self.phoneme_to_id)
                self.phoneme_to_id[phoneme] = idx
                self.id_to_phoneme[idx] = phoneme
            
            logger.info(f"Phoneme vocabulary size: {len(self.phoneme_to_id)}")
            logger.info(f"Unique MFA phonemes: {len(phonemes_sorted)}")
            
            # Save vocabulary for val split
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                vocab_file = self.cache_dir / "phoneme_vocab.npy"
                np.save(vocab_file, {
                    'phoneme_to_id': self.phoneme_to_id,
                    'id_to_phoneme': self.id_to_phoneme
                })
                logger.info(f"Saved vocabulary to {vocab_file}")
        else:
            # Load vocabulary from training
            if cache_dir:
                vocab_file = Path(cache_dir) / "phoneme_vocab.npy"
                if vocab_file.exists():
                    vocab_data = np.load(vocab_file, allow_pickle=True).item()
                    self.phoneme_to_id = vocab_data['phoneme_to_id']
                    self.id_to_phoneme = vocab_data['id_to_phoneme']
                    logger.info(f"Loaded vocabulary from {vocab_file}")
                else:
                    raise FileNotFoundError(
                        f"Vocabulary file not found: {vocab_file}. Train split must be created first."
                    )
            else:
                raise ValueError("cache_dir required for validation split to load vocabulary")
        
        # Filter samples by max length if specified
        if self.max_phoneme_length:
            original_len = len(self.sample_ids)
            self.sample_ids = [
                sid for sid in self.sample_ids
                if len(self.alignments[sid]) <= self.max_phoneme_length
            ]
            logger.info(
                f"Filtered {original_len - len(self.sample_ids)} samples exceeding "
                f"max length {self.max_phoneme_length}"
            )
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load LJSpeech metadata.csv."""
        metadata_file = self.ljspeech_dir / "metadata.csv"
        metadata = {}
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    file_id = parts[0]
                    text = parts[2]  # Normalized text
                    metadata[file_id] = {
                        'text': text,
                        'audio_path': self.ljspeech_dir / 'wavs' / f'{file_id}.wav'
                    }
        
        return metadata
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a single training example.
        
        Returns:
            Dictionary containing:
                - phoneme_ids: [seq_len] Phoneme ID sequence
                - durations: [seq_len] Duration in frames per phoneme
                - text: Original text string
                - file_id: Sample identifier
        """
        sample_id = self.sample_ids[idx]
        
        # Get text (for reference only)
        text = self.metadata[sample_id]['text']
        
        # Get phonemes and durations from MFA alignments
        # Use MFA's phoneme sequence directly to ensure consistency
        alignment_phones = self.alignments[sample_id]
        
        # Extract phoneme sequence and durations
        phoneme_strings = []
        durations = []
        
        for phone_info in alignment_phones:
            phone = phone_info['phone']
            
            # Convert duration from seconds to frames
            duration_sec = phone_info['duration']
            duration_frames = int(np.round(duration_sec * self.sample_rate / self.hop_length))
            # Ensure at least 1 frame
            duration_frames = max(1, duration_frames)
            
            phoneme_strings.append(phone)
            durations.append(duration_frames)
        
        # Convert phonemes to IDs using vocabulary
        phoneme_ids = []
        for phone in phoneme_strings:
            # Map to ID, use <UNK> token if not in vocab
            if phone in self.phoneme_to_id:
                phoneme_ids.append(self.phoneme_to_id[phone])
            else:
                # Use unknown token
                phoneme_ids.append(self.phoneme_to_id.get('<UNK>', 0))
                logger.debug(f"Unknown phoneme '{phone}' in {sample_id}, using <UNK>")
        
        phoneme_ids = np.array(phoneme_ids, dtype=np.int32)
        durations = np.array(durations, dtype=np.float32)
        
        return {
            'phoneme_ids': phoneme_ids,
            'durations': durations,
            'text': text,
            'file_id': sample_id,
            'length': len(phoneme_ids)
        }
    
    def get_vocab_size(self) -> int:
        """Get size of phoneme vocabulary."""
        return len(self.phoneme_to_id)
    
    def get_phoneme_to_id(self) -> Dict[str, int]:
        """Get phoneme to ID mapping."""
        return self.phoneme_to_id
    
    def get_id_to_phoneme(self) -> Dict[int, str]:
        """Get ID to phoneme mapping."""
        return self.id_to_phoneme


class LJSpeechVAEDataset:
    """
    Dataset for VAE training with mel-spectrograms and text conditioning.
    
    Loads:
    - Phoneme sequences (for encoder text conditioning)
    - Mel-spectrograms (for VAE input/target)
    - Durations (for length regulation)
    """
    
    def __init__(
        self,
        ljspeech_dir: str,
        alignments_dir: str,
        split: str = 'train',
        val_split: float = 0.05,
        sample_rate: int = 22050,
        hop_length: int = 256,
        n_mels: int = 80,
        max_frames: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize LJSpeech VAE dataset.
        
        Args:
            ljspeech_dir: Path to LJSpeech-1.1 directory
            alignments_dir: Path to MFA alignments
            split: 'train' or 'val'
            val_split: Validation split fraction
            sample_rate: Audio sample rate
            hop_length: STFT hop length
            n_mels: Number of mel bands
            max_frames: Maximum mel-spectrogram frames (for filtering)
            cache_dir: Cache directory
        """
        from iris.alignment import MFAAligner
        from iris.data import load_audio, compute_mel_spectrogram
        
        self.ljspeech_dir = Path(ljspeech_dir)
        self.alignments_dir = Path(alignments_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load metadata
        logger.info("Loading LJSpeech metadata...")
        self.metadata = self._load_metadata()
        
        # Load MFA alignments
        logger.info(f"Loading MFA alignments from {alignments_dir}...")
        aligner = MFAAligner()
        self.alignments = aligner.load_alignments(str(self.alignments_dir))
        logger.info(f"Loaded {len(self.alignments)} alignments")
        
        # Filter to samples with both
        valid_ids = set(self.metadata.keys()) & set(self.alignments.keys())
        logger.info(f"Found {len(valid_ids)} samples with metadata and alignments")
        
        # Split train/val
        all_ids = sorted(valid_ids)
        val_size = int(len(all_ids) * val_split)
        
        if split == 'val':
            self.sample_ids = all_ids[:val_size]
        else:
            self.sample_ids = all_ids[val_size:]
        
        logger.info(f"{split} split: {len(self.sample_ids)} samples")
        
        # Load or build phoneme vocabulary
        if split == 'train':
            logger.info("Building phoneme vocabulary from MFA alignments...")
            all_phonemes = set()
            for sid in self.sample_ids:
                for phone_info in self.alignments[sid]:
                    all_phonemes.add(phone_info['phone'])
            
            phonemes_sorted = sorted(all_phonemes)
            self.phoneme_to_id = {}
            self.id_to_phoneme = {}
            
            special_tokens = ['<PAD>', '<UNK>']
            for i, token in enumerate(special_tokens):
                self.phoneme_to_id[token] = i
                self.id_to_phoneme[i] = token
            
            for phoneme in phonemes_sorted:
                idx = len(self.phoneme_to_id)
                self.phoneme_to_id[phoneme] = idx
                self.id_to_phoneme[idx] = phoneme
            
            logger.info(f"Phoneme vocabulary size: {len(self.phoneme_to_id)}")
            
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                vocab_file = self.cache_dir / "phoneme_vocab.npy"
                np.save(vocab_file, {
                    'phoneme_to_id': self.phoneme_to_id,
                    'id_to_phoneme': self.id_to_phoneme
                })
                logger.info(f"Saved vocabulary to {vocab_file}")
        else:
            if cache_dir:
                vocab_file = Path(cache_dir) / "phoneme_vocab.npy"
                if vocab_file.exists():
                    vocab_data = np.load(vocab_file, allow_pickle=True).item()
                    self.phoneme_to_id = vocab_data['phoneme_to_id']
                    self.id_to_phoneme = vocab_data['id_to_phoneme']
                    logger.info(f"Loaded vocabulary from {vocab_file}")
                else:
                    raise FileNotFoundError(f"Vocabulary not found: {vocab_file}")
            else:
                raise ValueError("cache_dir required for val split")
        
        # Filter by max frames if specified
        if self.max_frames:
            original_len = len(self.sample_ids)
            # Estimate frames from duration
            self.sample_ids = [
                sid for sid in self.sample_ids
                if self._estimate_frames(sid) <= self.max_frames
            ]
            logger.info(
                f"Filtered {original_len - len(self.sample_ids)} samples "
                f"exceeding max_frames {self.max_frames}"
            )
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load LJSpeech metadata."""
        metadata_file = self.ljspeech_dir / "metadata.csv"
        metadata = {}
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    file_id = parts[0]
                    text = parts[2]
                    metadata[file_id] = {
                        'text': text,
                        'audio_path': self.ljspeech_dir / 'wavs' / f'{file_id}.wav'
                    }
        
        return metadata
    
    def _estimate_frames(self, sample_id: str) -> int:
        """Estimate number of mel-spec frames from alignments."""
        alignment_phones = self.alignments[sample_id]
        total_duration = sum(p['duration'] for p in alignment_phones)
        frames = int(total_duration * self.sample_rate / self.hop_length)
        return frames
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get training sample.
        
        Returns:
            - phoneme_ids: [seq_len] Phoneme IDs
            - mel_spec: [n_mels, time] Mel-spectrogram
            - durations: [seq_len] Durations in frames
            - text: Text string
            - file_id: Sample ID
        """
        from iris.data import load_audio, compute_mel_spectrogram
        
        sample_id = self.sample_ids[idx]
        
        # Get text
        text = self.metadata[sample_id]['text']
        
        # Get phonemes and durations from MFA
        alignment_phones = self.alignments[sample_id]
        
        phoneme_strings = []
        durations = []
        
        for phone_info in alignment_phones:
            phone = phone_info['phone']
            duration_sec = phone_info['duration']
            duration_frames = int(np.round(duration_sec * self.sample_rate / self.hop_length))
            duration_frames = max(1, duration_frames)
            
            phoneme_strings.append(phone)
            durations.append(duration_frames)
        
        # Convert to IDs
        phoneme_ids = []
        for phone in phoneme_strings:
            if phone in self.phoneme_to_id:
                phoneme_ids.append(self.phoneme_to_id[phone])
            else:
                phoneme_ids.append(self.phoneme_to_id.get('<UNK>', 0))
        
        phoneme_ids = np.array(phoneme_ids, dtype=np.int32)
        durations = np.array(durations, dtype=np.float32)
        
        # Load audio and compute mel-spectrogram
        audio_path = self.metadata[sample_id]['audio_path']
        audio = load_audio(str(audio_path), sample_rate=self.sample_rate)
        
        mel_spec = compute_mel_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        return {
            'phoneme_ids': phoneme_ids,
            'mel_spec': mel_spec,  # [n_mels, time]
            'durations': durations,
            'text': text,
            'file_id': sample_id,
            'phoneme_length': len(phoneme_ids),
            'mel_length': mel_spec.shape[1]
        }
    
    def get_vocab_size(self) -> int:
        return len(self.phoneme_to_id)
    
    def get_phoneme_to_id(self) -> Dict[str, int]:
        return self.phoneme_to_id
    
    def get_id_to_phoneme(self) -> Dict[int, str]:
        return self.id_to_phoneme


def collate_duration_batch(batch: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Collate function for batching variable-length sequences.
    
    Pads sequences to the maximum length in the batch.
    
    Args:
        batch: List of samples from LJSpeechDurationDataset
        
    Returns:
        Dictionary containing batched and padded tensors:
            - phoneme_ids: [batch, max_len] Padded phoneme IDs
            - durations: [batch, max_len] Padded durations
            - lengths: [batch] Actual sequence lengths
            - texts: List of text strings
            - file_ids: List of file identifiers
    """
    # Get max length in batch
    max_len = max(item['length'] for item in batch)
    batch_size = len(batch)
    
    # Initialize padded arrays
    phoneme_ids = np.zeros((batch_size, max_len), dtype=np.int32)
    durations = np.zeros((batch_size, max_len), dtype=np.float32)
    lengths = np.array([item['length'] for item in batch], dtype=np.int32)
    texts = [item['text'] for item in batch]
    file_ids = [item['file_id'] for item in batch]
    
    # Fill in actual values
    for i, item in enumerate(batch):
        seq_len = item['length']
        phoneme_ids[i, :seq_len] = item['phoneme_ids']
        durations[i, :seq_len] = item['durations']
    
    return {
        'phoneme_ids': phoneme_ids,
        'durations': durations,
        'lengths': lengths,
        'texts': texts,
        'file_ids': file_ids,
    }


def collate_vae_batch(batch: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Collate function for VAE training batches.
    
    Pads both phoneme sequences and mel-spectrograms.
    
    Args:
        batch: List of samples from LJSpeechVAEDataset
        
    Returns:
        Dictionary containing batched and padded tensors
    """
    batch_size = len(batch)
    
    # Get max lengths
    max_phoneme_len = max(item['phoneme_length'] for item in batch)
    max_mel_len = max(item['mel_length'] for item in batch)
    n_mels = batch[0]['mel_spec'].shape[0]
    
    # Initialize padded arrays
    phoneme_ids = np.zeros((batch_size, max_phoneme_len), dtype=np.int32)
    mel_specs = np.zeros((batch_size, n_mels, max_mel_len), dtype=np.float32)
    durations = np.zeros((batch_size, max_phoneme_len), dtype=np.float32)
    
    phoneme_lengths = np.array([item['phoneme_length'] for item in batch], dtype=np.int32)
    mel_lengths = np.array([item['mel_length'] for item in batch], dtype=np.int32)
    
    texts = [item['text'] for item in batch]
    file_ids = [item['file_id'] for item in batch]
    
    # Fill in actual values
    for i, item in enumerate(batch):
        p_len = item['phoneme_length']
        m_len = item['mel_length']
        
        phoneme_ids[i, :p_len] = item['phoneme_ids']
        mel_specs[i, :, :m_len] = item['mel_spec']
        durations[i, :p_len] = item['durations']
    
    return {
        'phoneme_ids': phoneme_ids,
        'mel_specs': mel_specs,  # [batch, n_mels, time]
        'durations': durations,
        'phoneme_lengths': phoneme_lengths,
        'mel_lengths': mel_lengths,
        'texts': texts,
        'file_ids': file_ids,
    }

