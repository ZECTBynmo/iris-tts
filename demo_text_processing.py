#!/usr/bin/env python3
"""
Demo script for text processing with CMUDict and g2p_en.

Usage:
    python demo_text_processing.py
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 70)
    logger.info("Text Processing Demo - CMUDict + g2p_en")
    logger.info("=" * 70)
    
    # Step 1: Initialize text processor
    logger.info("\n[1/5] Initializing text processor...")
    
    try:
        from iris.text import create_text_processor
        processor = create_text_processor(use_g2p=True)
    except Exception as e:
        logger.error(f"Failed to initialize text processor: {e}")
        logger.info("\nMake sure to install dependencies:")
        logger.info("  uv sync")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Step 2: Test text normalization
    logger.info("\n[2/5] Testing text normalization...")
    
    # Basic normalization tests
    logger.info("\n  Basic normalization:")
    basic_texts = [
        "Hello, World!",
        "THE QUICK BROWN FOX",
        "Text   with    extra    spaces",
    ]
    
    for text in basic_texts:
        normalized = processor.normalize_text(text)
        logger.info(f"    '{text}' -> '{normalized}'")
    
    # NeMo normalization tests (numbers, dates, etc.)
    logger.info("\n  NeMo advanced normalization:")
    nemo_texts = [
        "I have $123.45 in my account",
        "The meeting is on 12/25/2023 at 3:30pm",
        "Call me at 555-1234",
        "The temperature is 72°F",
        "He was born in 1990",
        "She ran 26.2 miles",
    ]
    
    for text in nemo_texts:
        try:
            normalized = processor.normalize_text(text)
            logger.info(f"    '{text}'")
            logger.info(f"      -> '{normalized}'")
        except Exception as e:
            logger.warning(f"    '{text}' -> Error: {e}")
    
    # Step 3: Test phoneme conversion
    logger.info("\n[3/5] Testing text-to-phoneme conversion...")
    
    sentences = [
        "Hello world",
        "This is a test",
        "The quick brown fox jumps over the lazy dog",
        "Synthesize speech from text",
    ]
    
    for sentence in sentences:
        phonemes = processor.text_to_phonemes(sentence)
        logger.info(f"\n  Text: {sentence}")
        logger.info(f"  Phonemes: {phonemes}")
    
    # Step 4: Test CMUDict vs g2p
    logger.info("\n[4/5] Testing CMUDict vs g2p_en...")
    
    # Words in CMUDict
    common_words = ["hello", "world", "speech", "synthesis"]
    logger.info("\n  Common words (from CMUDict):")
    for word in common_words:
        phonemes = processor.word_to_phonemes(word)
        logger.info(f"    {word} -> {' '.join(phonemes)}")
    
    # Out-of-vocabulary words (will use g2p)
    oov_words = ["tensorflow", "pytorch", "keras", "neuralnet"]
    logger.info("\n  Out-of-vocabulary words (from g2p_en):")
    for word in oov_words:
        phonemes = processor.word_to_phonemes(word)
        logger.info(f"    {word} -> {' '.join(phonemes)}")
    
    # Step 5: Create phoneme vocabulary from LJSpeech
    logger.info("\n[5/5] Creating phoneme vocabulary from LJSpeech...")
    
    metadata_file = Path("data/LJSpeech-1.1/metadata.csv")
    
    if metadata_file.exists():
        logger.info(f"  Reading LJSpeech metadata from {metadata_file}")
        
        # Read first 100 lines of metadata
        texts = []
        with open(metadata_file, "r") as f:
            for i, line in enumerate(f):
                if i >= 100:  # Just use first 100 for demo
                    break
                parts = line.strip().split("|")
                if len(parts) >= 3:
                    # Use normalized text (column 3)
                    texts.append(parts[2])
        
        logger.info(f"  Loaded {len(texts)} text samples")
        
        # Get unique phonemes
        phoneme_set = processor.get_phoneme_set(texts)
        logger.info(f"  Found {len(phoneme_set)} unique phonemes")
        logger.info(f"  Phonemes: {sorted(phoneme_set)}")
        
        # Create phoneme mappings
        phoneme_to_id, id_to_phoneme = processor.create_phoneme_mapping(texts)
        logger.info(f"\n  Created phoneme vocabulary with {len(phoneme_to_id)} entries")
        logger.info(f"  Special tokens: {[id_to_phoneme[i] for i in range(4)]}")
        
        # Test sequence conversion
        test_text = "Hello world"
        phonemes = processor.text_to_phonemes(test_text)
        sequence = processor.text_to_sequence(test_text, phoneme_to_id)
        logger.info(f"\n  Example conversion:")
        logger.info(f"    Text: {test_text}")
        logger.info(f"    Phonemes: {phonemes}")
        logger.info(f"    Sequence IDs: {sequence}")
    else:
        logger.warning(f"  LJSpeech metadata not found at {metadata_file}")
        logger.info("  Skipping vocabulary creation demo")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✓ Demo complete!")
    logger.info("=" * 70)
    
    logger.info("\nText Processing Pipeline Features:")
    logger.info("  ✓ NeMo text normalization (numbers, dates, currencies)")
    logger.info("  ✓ CMUDict phoneme lookup")
    logger.info("  ✓ g2p_en for OOV words")
    logger.info("  ✓ Phoneme vocabulary creation")
    logger.info("  ✓ Text-to-sequence conversion")
    
    # Usage example
    logger.info("\n" + "=" * 70)
    logger.info("Usage in your TTS pipeline:")
    logger.info("=" * 70)
    print("""
from iris.text import create_text_processor

# Initialize processor with NeMo normalization
processor = create_text_processor(use_g2p=True, use_nemo=True)

# Normalize text (handles numbers, dates, etc.)
text = "I have $50 and the meeting is at 3:30pm"
normalized = processor.normalize_text(text)
print(f"Normalized: {normalized}")

# Convert text to phonemes
text = "Hello world, this is a test"
phonemes = processor.text_to_phonemes(text)
print(f"Phonemes: {phonemes}")

# Create vocabulary from your dataset
texts = ["text1", "text2", "text3", ...]
phoneme_to_id, id_to_phoneme = processor.create_phoneme_mapping(texts)

# Convert text to sequence of IDs
sequence = processor.text_to_sequence(text, phoneme_to_id)
print(f"Sequence: {sequence}")
""")


if __name__ == "__main__":
    main()

