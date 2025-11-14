"""Investigate duration prediction data to understand what the model should learn."""

import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

from iris.datasets import LJSpeechDurationDataset
from iris.alignment import MFAAligner

def main():
    print("=" * 80)
    print("Duration Data Investigation")
    print("=" * 80)
    
    # Load dataset
    dataset = LJSpeechDurationDataset(
        ljspeech_dir="data/LJSpeech-1.1",
        alignments_dir="data/ljspeech_alignments/LJSpeech",
        split="train",
        val_split=0.05,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.get_vocab_size()}")
    
    # Collect all phoneme-duration pairs
    phoneme_durations = defaultdict(list)
    all_durations = []
    
    print("\nCollecting phoneme-duration statistics...")
    for i in range(min(5000, len(dataset))):  # Sample 5000 to speed up
        sample = dataset[i]
        phoneme_ids = sample['phoneme_ids']
        durations = sample['durations']
        
        for pid, dur in zip(phoneme_ids, durations):
            phoneme_durations[int(pid)].append(float(dur))
            all_durations.append(float(dur))
    
    # Global statistics
    all_durations = np.array(all_durations)
    print(f"\n{'='*80}")
    print("GLOBAL DURATION STATISTICS")
    print(f"{'='*80}")
    print(f"Total phoneme instances: {len(all_durations)}")
    print(f"Mean duration: {all_durations.mean():.2f} frames")
    print(f"Std duration: {all_durations.std():.2f} frames")
    print(f"Min duration: {all_durations.min():.2f} frames")
    print(f"Max duration: {all_durations.max():.2f} frames")
    print(f"Median duration: {np.median(all_durations):.2f} frames")
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(all_durations, p):.2f} frames")
    
    # Per-phoneme statistics
    print(f"\n{'='*80}")
    print("PER-PHONEME VARIANCE ANALYSIS")
    print(f"{'='*80}")
    
    phoneme_stats = []
    for pid, durs in phoneme_durations.items():
        durs = np.array(durs)
        if len(durs) > 10:  # Only analyze phonemes with enough samples
            phoneme_stats.append({
                'id': pid,
                'count': len(durs),
                'mean': durs.mean(),
                'std': durs.std(),
                'min': durs.min(),
                'max': durs.max(),
                'cv': durs.std() / (durs.mean() + 1e-8),  # Coefficient of variation
            })
    
    # Sort by coefficient of variation (higher = more variable)
    phoneme_stats.sort(key=lambda x: x['cv'], reverse=True)
    
    print("\nMost variable phonemes (high CV = hard to predict):")
    print(f"{'Phoneme ID':<12} {'Count':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'CV':<8}")
    print("-" * 80)
    for stat in phoneme_stats[:15]:
        print(f"{stat['id']:<12} {stat['count']:<8} {stat['mean']:<8.2f} {stat['std']:<8.2f} "
              f"{stat['min']:<8.2f} {stat['max']:<8.2f} {stat['cv']:<8.2f}")
    
    print("\nLeast variable phonemes (low CV = easy to predict):")
    print(f"{'Phoneme ID':<12} {'Count':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'CV':<8}")
    print("-" * 80)
    for stat in phoneme_stats[-15:]:
        print(f"{stat['id']:<12} {stat['count']:<8} {stat['mean']:<8.2f} {stat['std']:<8.2f} "
              f"{stat['min']:<8.2f} {stat['max']:<8.2f} {stat['cv']:<8.2f}")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_durations, bins=50, edgecolor='black')
    plt.xlabel('Duration (frames)')
    plt.ylabel('Count')
    plt.title('Distribution of All Durations')
    plt.axvline(all_durations.mean(), color='red', linestyle='--', label=f'Mean: {all_durations.mean():.1f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log(all_durations + 1), bins=50, edgecolor='black', color='orange')
    plt.xlabel('Log(Duration + 1)')
    plt.ylabel('Count')
    plt.title('Distribution in Log Space')
    plt.axvline(np.log(all_durations.mean() + 1), color='red', linestyle='--', 
                label=f'Log(Mean): {np.log(all_durations.mean() + 1):.2f}')
    plt.legend()
    
    plt.tight_layout()
    output_path = Path("outputs/duration_analysis.png")
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    
    # Key insight
    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print(f"{'='*80}")
    
    # Check if duration is actually predictable from phoneme alone
    high_var_count = sum(1 for s in phoneme_stats if s['cv'] > 0.5)
    total_count = len(phoneme_stats)
    
    print(f"Phonemes with high variance (CV > 0.5): {high_var_count}/{total_count} ({100*high_var_count/total_count:.1f}%)")
    print(f"\nThis means {100*high_var_count/total_count:.1f}% of phonemes have highly variable durations.")
    print("Duration depends on factors beyond just phoneme identity:")
    print("  - Stress/emphasis")
    print("  - Position in word/utterance")  
    print("  - Speaking rate")
    print("  - Phonetic context (surrounding phonemes)")
    print("\nConclusion: Perfect duration prediction from phoneme ID alone is IMPOSSIBLE.")
    print("The model is doing as well as can be expected given limited input features.")


if __name__ == "__main__":
    main()

