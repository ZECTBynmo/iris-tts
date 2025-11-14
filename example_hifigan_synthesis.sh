#!/bin/bash
# Example: Synthesize speech using the pre-trained HiFiGAN vocoder

# Make sure you're in the iris directory with conda and venv activated
# conda activate iris-tts
# source .venv/bin/activate

# Synthesize with HiFiGAN vocoder
uv run python scripts/synthesize.py \
  --text "Hello world, this is a test of the HiFiGAN vocoder." \
  --output_wav "outputs/sample_hifigan.wav" \
  --vocoder hifigan \
  --vocoder_entry iris.hifigan_pretrained:infer_hifigan \
  --vocab_path "outputs/vae/cache/phoneme_vocab.npy" \
  --encoder_weights "outputs/encoder/checkpoints/encoder_best.weights.h5" \
  --duration_weights "outputs/encoder/checkpoints/duration_best.weights.h5" \
  --vae_core_weights "outputs/vae/checkpoints_vae/vae_core_best.weights.h5" \
  --postnet_weights "outputs/vae/checkpoints_postnet/postnet_best.weights.h5"

echo ""
echo "Audio generated with HiFiGAN vocoder at: outputs/sample_hifigan.wav"
echo "Compare with Griffin-Lim by running without --vocoder_entry flag"

