"""Text processing utilities using CMUDict, g2p_en, and NeMo text normalization."""

import re
import logging
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import nltk

logger = logging.getLogger(__name__)

# Try to import g2p_en
try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    logger.warning("g2p_en not available. Install with: pip install g2p-en")

# Try to import NeMo text processing
try:
    from nemo_text_processing.text_normalization.normalize import Normalizer
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("nemo_text_processing not available. Install with: pip install nemo-text-processing")


class TextProcessor:
    """
    Text processing pipeline using CMUDict, g2p_en, and NeMo text normalization.
    
    Features:
    - Advanced text normalization with NeMo (numbers, dates, currencies, etc.)
    - Word to phoneme conversion using CMUDict
    - Grapheme-to-phoneme (G2P) for out-of-vocabulary words
    - Special handling for numbers, abbreviations, etc.
    """
    
    def __init__(self, use_g2p: bool = True, use_nemo: bool = True, lang: str = "en"):
        """
        Initialize text processor.
        
        Args:
            use_g2p: Whether to use g2p_en for OOV words (default: True)
            use_nemo: Whether to use NeMo text normalization (default: True)
            lang: Language code for NeMo normalization (default: "en")
        """
        self.use_g2p = use_g2p and G2P_AVAILABLE
        self.use_nemo = use_nemo and NEMO_AVAILABLE
        self.lang = lang
        
        # Download CMUDict if not already available
        self._setup_cmudict()
        
        # Initialize g2p if available
        if self.use_g2p:
            logger.info("Initializing g2p_en...")
            self.g2p = G2p()
            logger.info("g2p_en initialized")
        else:
            self.g2p = None
            if use_g2p:
                logger.warning("g2p_en requested but not available")
        
        # Initialize NeMo normalizer if available
        if self.use_nemo:
            logger.info("Initializing NeMo text normalizer...")
            try:
                self.normalizer = Normalizer(
                    input_case="cased",
                    lang=lang,
                )
                logger.info("NeMo text normalizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NeMo normalizer: {e}")
                self.normalizer = None
                self.use_nemo = False
        else:
            self.normalizer = None
            if use_nemo:
                logger.warning("NeMo text normalization requested but not available")
    
    def _setup_cmudict(self):
        """Download and setup CMUDict."""
        try:
            self.cmudict = nltk.corpus.cmudict.dict()
            logger.info(f"CMUDict loaded with {len(self.cmudict)} entries")
        except LookupError:
            logger.info("Downloading CMUDict...")
            nltk.download('cmudict', quiet=True)
            self.cmudict = nltk.corpus.cmudict.dict()
            logger.info(f"CMUDict downloaded and loaded with {len(self.cmudict)} entries")
    
    def normalize_text(self, text: str, use_nemo: Optional[bool] = None) -> str:
        """
        Normalize text for TTS processing.
        
        Uses NeMo text normalization if available for advanced normalization
        (numbers, dates, currencies, etc.), otherwise falls back to basic normalization.
        
        Args:
            text: Input text string
            use_nemo: Override instance setting for NeMo usage (optional)
            
        Returns:
            Normalized text
        """
        # Determine whether to use NeMo
        use_nemo_norm = self.use_nemo if use_nemo is None else (use_nemo and self.normalizer is not None)
        
        if use_nemo_norm and self.normalizer:
            try:
                # NeMo normalization handles numbers, dates, currencies, etc.
                text = self.normalizer.normalize(text, verbose=False)
            except Exception as e:
                logger.warning(f"NeMo normalization failed, using basic normalization: {e}")
                use_nemo_norm = False
        
        if not use_nemo_norm:
            # Basic normalization fallback
            # Convert to lowercase
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def text_to_phonemes(self, text: str, separator: str = " ") -> str:
        """
        Convert text to phoneme sequence.
        
        Args:
            text: Input text string
            separator: Separator between phonemes (default: space)
            
        Returns:
            Phoneme sequence as string
        """
        # Normalize text
        text = self.normalize_text(text)
        
        # Split into words
        words = text.split()
        
        phoneme_sequence = []
        
        for word in words:
            # Remove punctuation from word
            word_clean = re.sub(r'[^\w]', '', word)
            
            if not word_clean:
                continue
            
            # Get phonemes for word
            phonemes = self.word_to_phonemes(word_clean)
            phoneme_sequence.extend(phonemes)
        
        return separator.join(phoneme_sequence)
    
    def word_to_phonemes(self, word: str) -> List[str]:
        """
        Convert a single word to phonemes.
        
        Args:
            word: Single word (no spaces or punctuation)
            
        Returns:
            List of phonemes
        """
        word_lower = word.lower()
        
        # Try CMUDict first
        if word_lower in self.cmudict:
            # CMUDict returns list of possible pronunciations
            # We take the first one
            phonemes = self.cmudict[word_lower][0]
            # Remove stress markers (0, 1, 2) from vowels
            phonemes = [self._remove_stress(p) for p in phonemes]
            return phonemes
        
        # Fall back to g2p for OOV words
        if self.use_g2p and self.g2p:
            phonemes = self.g2p(word)
            # g2p_en returns phonemes with some formatting, clean them up
            phonemes = [p for p in phonemes if p not in [' ', '']]
            return phonemes
        
        # If no g2p, return characters as fallback
        logger.warning(f"Word '{word}' not in CMUDict and g2p not available, using characters")
        return list(word_lower)
    
    def _remove_stress(self, phoneme: str) -> str:
        """Remove stress markers from phonemes."""
        return re.sub(r'[0-2]', '', phoneme)
    
    def text_to_sequence(
        self, 
        text: str, 
        phoneme_to_id: Optional[Dict[str, int]] = None
    ) -> List[int]:
        """
        Convert text to sequence of phoneme IDs.
        
        Args:
            text: Input text string
            phoneme_to_id: Dictionary mapping phonemes to integer IDs.
                          If None, returns phoneme strings.
            
        Returns:
            List of phoneme IDs
        """
        phonemes = self.text_to_phonemes(text).split()
        
        if phoneme_to_id is None:
            logger.warning("No phoneme_to_id mapping provided, returning phoneme strings")
            return phonemes
        
        sequence = []
        for phoneme in phonemes:
            if phoneme in phoneme_to_id:
                sequence.append(phoneme_to_id[phoneme])
            else:
                logger.warning(f"Unknown phoneme '{phoneme}', skipping")
        
        return sequence
    
    def get_phoneme_set(self, texts: List[str]) -> set:
        """
        Get set of unique phonemes from a list of texts.
        
        Useful for building phoneme vocabulary.
        
        Args:
            texts: List of text strings
            
        Returns:
            Set of unique phonemes
        """
        phonemes = set()
        for text in texts:
            text_phonemes = self.text_to_phonemes(text).split()
            phonemes.update(text_phonemes)
        
        return phonemes
    
    def create_phoneme_mapping(
        self, 
        texts: List[str],
        add_special_tokens: bool = True
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Create phoneme to ID and ID to phoneme mappings.
        
        Args:
            texts: List of texts to extract phonemes from
            add_special_tokens: Add special tokens (PAD, SOS, EOS, UNK)
            
        Returns:
            Tuple of (phoneme_to_id, id_to_phoneme) dictionaries
        """
        phonemes = sorted(self.get_phoneme_set(texts))
        
        phoneme_to_id = {}
        id_to_phoneme = {}
        
        current_id = 0
        
        # Add special tokens
        if add_special_tokens:
            special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
            for token in special_tokens:
                phoneme_to_id[token] = current_id
                id_to_phoneme[current_id] = token
                current_id += 1
        
        # Add phonemes
        for phoneme in phonemes:
            phoneme_to_id[phoneme] = current_id
            id_to_phoneme[current_id] = phoneme
            current_id += 1
        
        logger.info(f"Created phoneme mapping with {len(phoneme_to_id)} entries")
        
        return phoneme_to_id, id_to_phoneme


def create_text_processor(
    use_g2p: bool = True,
    use_nemo: bool = True,
    lang: str = "en"
) -> TextProcessor:
    """
    Create a text processor instance.
    
    Args:
        use_g2p: Whether to use g2p_en for OOV words
        use_nemo: Whether to use NeMo text normalization
        lang: Language code for NeMo normalization (default: "en")
        
    Returns:
        TextProcessor instance
    """
    return TextProcessor(use_g2p=use_g2p, use_nemo=use_nemo, lang=lang)

