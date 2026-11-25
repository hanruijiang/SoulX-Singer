#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m cli.inference_svc \
    --device "${DEVICE:-cuda}" \
    --model_path pretrained_models/SoulX-Singer/model-svc.pt \
    --prompt_wav_path example/audio/zh_prompt.mp3 \
    --target_wav_path example/audio/music.mp3 \
    --prompt_f0_path example/audio/zh_prompt_f0.npy \
    --target_f0_path example/audio/music_f0.npy \
    --save_dir example/generated/music_svc \
    --auto_shift "$@"
