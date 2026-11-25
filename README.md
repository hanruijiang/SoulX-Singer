# SoulX-Singer-SVC

Command-line singing voice conversion (SVC). Convert source vocals to a reference singer's timbre while preserving lyrics, melody and rhythm. F0 can be supplied as NumPy arrays or extracted automatically with RMVPE.

## Installation

Use Python 3.10 and a PyTorch/Torchaudio build suitable for your platform and CUDA version.

```bash
git clone -b refactor/svc-cli-only --single-branch https://github.com/hanruijiang/SoulX-Singer.git
cd SoulX-Singer
conda create -n soulxsinger-svc python=3.10 -y
conda activate soulxsinger-svc
pip install -r requirements.txt
```

## Model weights

Download the SVC checkpoint:

```bash
hf download Soul-AILab/SoulX-Singer model-svc.pt --local-dir pretrained_models/SoulX-Singer
```

For automatic F0 extraction, download RMVPE:

```bash
hf download Soul-AILab/SoulX-Singer-Preprocess rmvpe/rmvpe.pt --local-dir pretrained_models
```

Whisper Base is downloaded from `openai/whisper-base` on first model initialization. For offline use, cache it in advance. Use trusted model checkpoints.

## Inference

Run commands from the repository root. Use isolated vocals without accompaniment. The reference clip must be 0.1–30 seconds; the source clip must be at least 0.1 seconds. Audio is converted to mono and resampled to 24 kHz for SVC and 16 kHz for RMVPE. Long sources are converted in segments of at most 30 seconds; RMVPE extracts F0 from the complete input.

### Automatic F0

```bash
python -m cli.inference_svc \
  --prompt_wav_path /path/to/reference.wav \
  --target_wav_path /path/to/source.wav \
  --save_dir outputs \
  --device cuda --auto_shift
```

### Precomputed F0

```bash
bash example/infer_svc.sh
```

The example uses bundled audio and F0 arrays. Pass your own arrays with `--prompt_f0_path` and `--target_f0_path`. Each array contains F0 in Hz at 50 frames/s (20 ms hop), aligned to its audio, with unvoiced frames set to zero. RMVPE is used only for missing arrays.

The result is saved as `<save_dir>/generated.wav` at 24 kHz.

| Option | Description |
| --- | --- |
| `--model_path` | SVC checkpoint |
| `--config` | Model YAML (default: `soulxsinger/config/soulxsinger.yaml`) |
| `--rmvpe_path` | RMVPE checkpoint |
| `--device` | CUDA device (e.g. cuda:0) or cpu |
| `--auto_shift` | Match median pitch to the reference singer |
| `--pitch_shift` | Manual shift in semitones; nonzero overrides automatic shift |
| `--n_steps` | Flow matching steps (default 32) |
| `--cfg` | Classifier-free guidance scale (default 3) |
| `--fp16` | Mixed precision on CUDA |

The audio and model settings in the YAML are fixed by the checkpoint. Customize `infer.n_steps` and `infer.cfg` or override them on the command line.

```bash
python -m cli.inference_svc --help
```

## Attribution and license

Based on [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) ([paper](https://arxiv.org/abs/2602.07803), [model weights](https://huggingface.co/Soul-AILab/SoulX-Singer)). Acknowledgements: [Amphion](https://github.com/open-mmlab/Amphion), [RMVPE](https://github.com/Dream-High/RMVPE), and [Whisper](https://github.com/openai/whisper).

Licensed under [Apache 2.0](LICENSE).
