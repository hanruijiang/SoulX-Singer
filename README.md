# SoulX-Singer-SVC

CLI-only singing voice conversion (SVC) based on [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer). Convert source singing to a reference singer's timbre while preserving lyrics, melody and rhythm. This repository retains only SVC inference and optional pitch extraction. It does not include a web interface or a full preprocessing pipeline.

## Installation

Python 3.10 is recommended. Install PyTorch/Torchaudio for your platform and CUDA version as appropriate; the requirements retain the original repository's Torch/Transformers version pair.

```bash
git clone https://github.com/hanruijiang/SoulX-Singer.git
cd SoulX-Singer
conda create -n soulxsinger-svc python=3.10 -y
conda activate soulxsinger-svc
pip install -r requirements.txt
```

## Download weights

Download only the SVC checkpoint (about 2.8 GB):

```bash
hf download Soul-AILab/SoulX-Singer model-svc.pt --local-dir pretrained_models/SoulX-Singer
```

Whisper Base is downloaded automatically from `openai/whisper-base` on first model initialization. For automatic F0 extraction, also download the RMVPE checkpoint (about 181 MB):

```bash
hf download Soul-AILab/SoulX-Singer-Preprocess rmvpe/rmvpe.pt --local-dir pretrained_models
```

No RMVPE weights are needed if both F0 arrays are supplied. Only load trusted checkpoints. For offline inference, cache Whisper and download the required weights beforehand.

## CLI inference

Run from the repository root. The prompt is a reference clip of the desired singer (0.1–30 seconds). The target is the source singing to convert; use isolated vocals without accompaniment. Input audio is converted to mono and resampled to 24 kHz. Long targets are segmented by the SVC model.

### Convert your own audio (automatic F0 extraction)

```bash
python -m cli.inference_svc \
  --prompt_wav_path /path/to/reference.wav \
  --target_wav_path /path/to/source.wav \
  --save_dir outputs \
  --device cuda --auto_shift
```

### Use precomputed F0

```bash
bash example/infer_svc.sh
```

The example uses the bundled audio and F0 arrays and does not require RMVPE. You can pass your own arrays with `--prompt_f0_path` and `--target_f0_path`. Each should be a 1-D NumPy array of F0 in Hz, sampled at 50 frames/s (20 ms hop), with unvoiced frames set to zero, aligned to the corresponding input audio. If one array is omitted, RMVPE extracts only that input.

The result is written to `<save_dir>/generated.wav` at 24 kHz.

| Option | Description |
| --- | --- |
| `--model_path` | SVC checkpoint path |
| `--config` | Model YAML (default: `soulxsinger/config/soulxsinger.yaml`) |
| `--rmvpe_path` | Optional pitch extractor checkpoint |
| `--device` | CUDA device (e.g. cuda:0) or cpu (slower) |
| `--auto_shift` | Match median pitch to the reference singer |
| `--pitch_shift` | Manual shift in semitones; nonzero overrides automatic shift |
| `--n_steps` | Flow matching steps (default 32) |
| `--cfg` | Classifier-free guidance scale (default 3) |
| `--fp16` | Half precision on CUDA |

```bash
python -m cli.inference_svc --help
```

## Attribution and license

Original model and inference implementation: [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer). See the [paper](https://arxiv.org/abs/2602.07803), [model weights](https://huggingface.co/Soul-AILab/SoulX-Singer) and [reference Space](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer). Acknowledgements: [Amphion](https://github.com/open-mmlab/Amphion), [RMVPE](https://github.com/Dream-High/RMVPE), and [Whisper](https://github.com/openai/whisper).

The original [Apache 2.0 license](LICENSE) and source attribution are retained. Use voices with permission; do not create deceptive impersonations.
