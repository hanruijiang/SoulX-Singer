"""Audio loading and resampling."""

import torchaudio


def load_wav(wav_path: str):
    """Load mono audio and return the waveform and native sample rate."""
    waveform, sample_rate = torchaudio.load(wav_path)
    return waveform.mean(dim=0, keepdim=True), sample_rate


def resample_wav(waveform, source_sr: int, target_sr: int):
    if source_sr == target_sr:
        return waveform
    return torchaudio.functional.resample(waveform, source_sr, target_sr)
