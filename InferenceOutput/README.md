# InferenceOutput — Curated Voice Samples

This directory holds shareable, version-controlled listening samples demonstrating the voice cloning capabilities of the fine-tuned `Qwen3-TTS-12Hz-0.6B-Base` LoRA model compared to the zero-shot baseline and original speaker reference.

## Published Benchmark Samples

| # | Sample File | Description | Source / Split |
|---|---|---|---|
| **01** | `01_original_voice_reference.wav` | Ground truth target speaker recording | Isolated reference split (`reference.wav`) |
| **02** | `02_zero_shot_baseline_sample.wav` | Qwen3-TTS 0.6B zero-shot baseline output | Base model holdout synthesis (`artifacts/zero_shot/`) |
| **03** | `03_qwen_finetuned_lora_windmill.wav` | LoRA-adapted checkpoint output (scale 0.3) | Fine-tuned holdout checkpoint (`artifacts/lora/`) |
| **04** | `04_qwen_finetuned_inference_technical.wav` | Out-of-domain technical vocabulary synthesis | Manual test prompt (Machine Learning / Deep Learning) |
| **05** | `05_qwen_finetuned_inference_narrative.wav` | Long-form complex narrative sentence | Manual test prompt (Cadence & breath pacing) |

## Exporting Samples from a Run

After completing a training run and evaluation pipeline (`bash scripts/train_and_eval.sh`), you can copy representative audio files into this directory:

```bash
# Copy reference audio
cp data/splits/reference.wav InferenceOutput/01_original_voice_reference.wav

# Copy zero-shot baseline sample
cp artifacts/zero_shot/zs_0000.wav InferenceOutput/02_zero_shot_baseline_sample.wav

# Copy fine-tuned LoRA candidate sample
cp artifacts/lora/lora_0000.wav InferenceOutput/03_qwen_finetuned_lora_windmill.wav
```

## Generating Waveform Videos for GitHub (Optional)

To convert audio samples to compact MP4 videos with animated waveforms (which render inline on GitHub):

```bash
# Requires ffmpeg
ffmpeg -y -i InferenceOutput/03_qwen_finetuned_lora_windmill.wav \
  -filter_complex "[0:a]showwaves=s=640x240:mode=line:colors=0x2563eb[v]" \
  -map "[v]" -map 0:a -c:v libx264 -c:a aac -b:a 192k -shortest \
  InferenceOutput/videos/03_qwen_finetuned_lora_windmill.mp4
```
