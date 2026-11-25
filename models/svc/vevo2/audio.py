# Copyright (c) 2023 Amphion. MIT license; see LICENSE.
from functools import lru_cache

import librosa
import numpy as np
import parselmouth
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from librosa.filters import mel as librosa_mel_fn

SAMPLE_RATE = 24000


@lru_cache(maxsize=4)
def read_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    if audio.size < SAMPLE_RATE // 10 or not np.isfinite(audio).all():
        raise ValueError(f"Audio must be finite and at least 0.1 seconds: {path}")
    return audio


def validate_audio_inputs(args):
    source = str(args.input)
    reference = str(args.reference) if getattr(args, "reference", None) else source
    timbre = str(args.timbre_reference) if getattr(args, "timbre_reference", None) else reference
    whisper_paths = {
        "convert": (source, reference),
        "tts": (source, timbre),
        "edit": (source,),
        "style": (source, reference),
        "melody": (reference, timbre),
    }[args.task]
    for path in set(whisper_paths):
        if read_audio(path).size > 30 * SAMPLE_RATE:
            raise ValueError(f"Whisper input exceeds 30 seconds: {path}")
    if args.task == "melody":
        read_audio(source)


def load_wav(path, device):
    speech = read_audio(path)
    speech24k = torch.tensor(speech).unsqueeze(0).to(device)
    speech16k = torchaudio.functional.resample(speech24k, SAMPLE_RATE, 16000)
    return speech, speech24k, speech16k


def save_audio(waveform, output_path, target_db=-25.0):
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise RuntimeError("Inference produced empty or non-finite audio")
    rms = torch.sqrt(torch.mean(waveform ** 2))
    current_db = 20 * torch.log10(rms + 1e-9)
    normalized = waveform * (10 ** ((target_db - current_db) / 20))
    sf.write(output_path, normalized.squeeze(0).cpu().numpy(), SAMPLE_RATE)


def extract_f0_hz(wav_path, fs=16000, hop_length=256, f0_min=50, f0_max=1100):
    audio, _ = librosa.load(wav_path, sr=fs)
    f0 = parselmouth.Sound(audio, fs).to_pitch_ac(
        time_step=hop_length / fs,
        voicing_threshold=0.6,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    ).selected_array["frequency"]
    voiced = f0 != 0
    if voiced.any():
        f0[~voiced] = np.interp(
            np.where(~voiced)[0], np.where(voiced)[0], f0[voiced]
        )
    return f0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        super(MelSpectrogram, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(win_size)

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("hann_window", hann_window)

    def forward(self, y):
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec)

        return spec
