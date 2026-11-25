"""Command-line inference for SoulX-Singer-SVC."""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def build_model(model_path, config, device="cuda", use_fp16=False):
    import torch
    from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC

    if not Path(model_path).is_file():
        raise FileNotFoundError(f"SVC checkpoint not found: {model_path}")
    model = SoulXSingerSVC(config)
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    if use_fp16 and torch.device(device).type == "cuda":
        # Keep mel analysis and the ISTFT window in their original float32 precision.
        model.f0_encoder.half()
        model.cfm_decoder.half()
        model.vocoder.model.backbone.half()
        model.vocoder.model.head.out.half()
    return model.eval().to(device)


def process(args, config, model=None):
    import numpy as np
    import soundfile as sf
    import torch
    from soulxsinger.utils.audio_utils import load_wav

    sample_rate = config.audio.sample_rate
    hop_size = config.audio.hop_size
    pt_wav = load_wav(args.prompt_wav_path, sample_rate)
    gt_wav = load_wav(args.target_wav_path, sample_rate)
    if pt_wav.shape[-1] < sample_rate // 10 or gt_wav.shape[-1] < sample_rate // 10:
        raise ValueError("Both audio inputs must be at least 0.1 seconds long.")
    if pt_wav.shape[-1] > 30 * sample_rate:
        raise ValueError("Prompt audio must be at most 30 seconds (Whisper context limit).")

    extractor = None
    if not args.prompt_f0_path or not args.target_f0_path:
        from soulxsinger.utils.f0_extraction import F0Extractor

        if not Path(args.rmvpe_path).is_file():
            raise FileNotFoundError(
                f"RMVPE checkpoint not found: {args.rmvpe_path}. "
                "Download it or supply both F0 .npy files."
            )
        extractor = F0Extractor(
            str(args.rmvpe_path), device=args.device,
            target_sr=sample_rate, hop_size=hop_size,
            max_duration=max(pt_wav.shape[-1], gt_wav.shape[-1]) / sample_rate + 1,
        )

    def get_f0(path, audio_path, waveform):
        f0 = np.load(path, allow_pickle=False) if path else extractor.process(audio_path)
        f0 = np.asarray(f0, dtype=np.float32).squeeze()
        if f0.ndim != 1 or not f0.size or not np.isfinite(f0).all() or (f0 < 0).any():
            raise ValueError(f"Expected a finite nonnegative F0 vector: {path or audio_path}")
        frames = math.ceil(waveform.shape[-1] / hop_size)
        if abs(len(f0) - frames) > 2:
            raise ValueError(
                f"F0 length {len(f0)} does not match {frames} audio frames: {path or audio_path}"
            )
        f0 = np.pad(f0, (0, max(0, frames - len(f0))))[:frames]
        if not (f0 > 0).any():
            raise ValueError(f"No voiced F0 frames: {path or audio_path}")
        return torch.from_numpy(f0).unsqueeze(0).to(args.device)

    pt_f0 = get_f0(args.prompt_f0_path, args.prompt_wav_path, pt_wav)
    gt_f0 = get_f0(args.target_f0_path, args.target_wav_path, gt_wav)
    del extractor
    use_fp16 = args.fp16
    if model is None:
        model = build_model(args.model_path, config, args.device, use_fp16)

    with torch.no_grad():
        audio, shift = model.infer(
            pt_wav=pt_wav.to(args.device),
            gt_wav=gt_wav.to(args.device),
            pt_f0=pt_f0, gt_f0=gt_f0,
            auto_shift=args.auto_shift, pitch_shift=args.pitch_shift,
            n_steps=args.n_steps if args.n_steps is not None else config.infer.n_steps,
            cfg=args.cfg if args.cfg is not None else config.infer.cfg,
            use_fp16=use_fp16,
        )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output = save_dir / "generated.wav"
    sf.write(output, audio.squeeze().float().cpu().numpy(), sample_rate)
    print(f"Generated audio: {output} (pitch shift: {shift} semitones)")



def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_path", default="pretrained_models/SoulX-Singer/model-svc.pt")
    parser.add_argument("--config", default="soulxsinger/config/soulxsinger.yaml")
    parser.add_argument("--prompt_wav_path", required=True, help="Reference singer audio (dry vocals)")
    parser.add_argument("--target_wav_path", required=True, help="Source singing audio (dry vocals)")
    parser.add_argument("--prompt_f0_path", help="Optional reference F0 .npy (Hz, 50 frames/s)")
    parser.add_argument("--target_f0_path", help="Optional source F0 .npy (Hz, 50 frames/s)")
    parser.add_argument("--rmvpe_path", default="pretrained_models/rmvpe/rmvpe.pt")
    parser.add_argument("--save_dir", default="outputs")
    parser.add_argument("--auto_shift", action="store_true")
    parser.add_argument("--pitch_shift", type=int, default=0)
    parser.add_argument("--n_steps", type=int, help="Flow matching steps (default: config)")
    parser.add_argument("--cfg", type=float, help="Guidance scale (default: config)")
    parser.add_argument("--fp16", action="store_true", help="Use half precision on CUDA")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.n_steps is not None and args.n_steps <= 0:
        parser.error("--n_steps must be positive")
    if args.cfg is not None and not math.isfinite(args.cfg):
        parser.error("--cfg must be finite")
    from soulxsinger.utils.file_utils import load_config

    process(args, load_config(args.config))
