# Copyright (c) 2023 Amphion. MIT license; see LICENSE.
import numpy as np
import librosa
import parselmouth
from models.svc.vevo2.config import JsonHParams

def interpolate(f0):
    """Interpolate the unvoiced part. Thus the f0 can be passed to a subtractive synthesizer.
    Args:
        f0: A numpy array of shape (seq_len,)
    Returns:
        f0: Interpolated f0 of shape (seq_len,)
        uv: Unvoiced part of shape (seq_len,)
    """
    uv = f0 == 0
    if len(f0[~uv]) > 0:
        # interpolate the unvoiced f0
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        uv = uv.astype("float")
        uv = np.min(np.array([uv[:-2], uv[1:-1], uv[2:]]), axis=0)
        uv = np.pad(uv, (1, 1))
    return f0, uv


def get_f0_features_using_parselmouth(audio, cfg, speed=1):
    """Using parselmouth to extract the f0 feature.
    Args:
        audio
        mel_len
        hop_length
        fs
        f0_min
        f0_max
        speed(default=1)
    Returns:
        f0: numpy array of shape (frame_len,)
        pitch_coarse: numpy array of shape (frame_len,)
    """
    hop_size = int(np.round(cfg.hop_size * speed))

    # Calculate the time step for pitch extraction
    time_step = hop_size / cfg.sample_rate * 1000

    f0 = (
        parselmouth.Sound(audio, cfg.sample_rate)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=0.6,
            pitch_floor=cfg.f0_min,
            pitch_ceiling=cfg.f0_max,
        )
        .selected_array["frequency"]
    )
    return f0


def extract_f0_hz(
    wav_path,
    fs=16000,
    hop_length=256,
    f0_min=50,
    f0_max=1100,
):
    cfg = JsonHParams()
    cfg.sample_rate = fs
    cfg.hop_size = hop_length
    cfg.f0_min = f0_min
    cfg.f0_max = f0_max

    # Compute f0
    audio, _ = librosa.load(wav_path, sr=fs)
    if audio.size < int(0.1 * fs) or not np.isfinite(audio).all():
        raise ValueError(f"Audio must be finite and at least 0.1 seconds: {wav_path}")
    f0 = get_f0_features_using_parselmouth(
        audio,
        cfg,
    )
    f0, _ = interpolate(f0)
    return f0


