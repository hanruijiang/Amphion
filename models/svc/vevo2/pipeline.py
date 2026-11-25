"""Vevo2 inference pipeline."""
import math
import re
import warnings

import numpy as np
import torch
import whisper
from librosa.effects import pitch_shift
from librosa.feature import chroma_stft
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from .audio import MelSpectrogram, extract_f0_hz, load_wav
from .checkpoint import (
    AR_DIR, CONTENT_DIR, FM_DIR, PROSODY_DIR,
    FM_TRAINING_PREFIXES, TOKENIZER_TRAINING_PREFIXES,
    load_weights, load_whisper, load_whisper_stats,
)
from .config import load_config
from .networks.flow_matching import FlowMatchingTransformer
from .networks.tokenizer import CocoContentStyle, CocoStyle
from .networks.vocos import Vocos
from .qwen_utils import gen_chat_prompt


def build_ar_model(path, device):
    kwargs = {
        "device_map": device,
        "torch_dtype": "auto",
        "local_files_only": True,
    }
    if (device.type == "cuda" and is_flash_attn_2_available()
            and torch.cuda.get_device_capability(device)[0] >= 8
            and torch.cuda.is_bf16_supported()):
        kwargs["attn_implementation"] = "flash_attention_2"
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()


class Vevo2InferencePipeline:
    def __init__(self, local, device, use_ar, use_prosody, local_files_only):
        self.device = device
        fm_cfg = load_config(local / FM_DIR / "config.json")
        self.fmt_cfg = fm_cfg
        self.fmt_model = load_weights(
            FlowMatchingTransformer(cfg=fm_cfg.model.flow_matching_transformer),
            local / FM_DIR / "model.safetensors", device, FM_TRAINING_PREFIXES,
        )
        self.whisper_model = load_whisper(device, local_files_only)
        self.whisper_mean, self.whisper_std = load_whisper_stats(
            local / FM_DIR / "whisper_stats.pt", device
        )
        self.content_style_tokenizer = load_weights(
            CocoContentStyle(cfg=fm_cfg.model.coco),
            local / CONTENT_DIR / "model.safetensors", device,
            TOKENIZER_TRAINING_PREFIXES,
        )
        if use_prosody:
            ar_cfg = load_config(local / AR_DIR / "amphion_config.json")
            self.style_tokenizer = load_weights(
                CocoStyle(cfg=ar_cfg.model.coco_style),
                local / PROSODY_DIR / "model.safetensors", device,
                TOKENIZER_TRAINING_PREFIXES,
            )
        vocoder_cfg = load_config(local / "vocoder/config.json")
        self.vocoder_cfg = vocoder_cfg
        cfg = vocoder_cfg.preprocess
        self.mel_model = MelSpectrogram(
            sampling_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            num_mels=cfg.num_mels, hop_size=cfg.hop_size,
            win_size=cfg.win_size, fmin=cfg.fmin, fmax=cfg.fmax,
        ).eval().to(device)
        self.vocoder_model = load_weights(
            Vocos(cfg=vocoder_cfg.model.vocos),
            local / "vocoder/model.safetensors", device,
        )
        if use_ar:
            self.ar_model = build_ar_model(local / AR_DIR, device)
            self.ar_tokenizer = AutoTokenizer.from_pretrained(
                local / AR_DIR, local_files_only=True,
            )

    @torch.no_grad()
    def extract_mel_feature(self, speech):
        mel_feature = self.mel_model(speech)  # (B, d, T)
        mel_feature = mel_feature.transpose(1, 2)
        mel_feature = (mel_feature - self.vocoder_cfg.preprocess.mel_mean) / math.sqrt(
            self.vocoder_cfg.preprocess.mel_var
        )
        return mel_feature

    @torch.no_grad()
    def extract_whisper_features(self, wavs, frame_lens):
        """
        Args:
            wavs: (B, T) at 16khz. Note that the max duration should be 30s
            frame_lens: (B,)
        Returns:
            features: (B, T, D)
        """
        # wavs: (batch, max_len)
        if wavs.shape[-1] > 30 * 16000:
            raise ValueError("Whisper feature extraction supports at most 30 seconds of audio; trim the input.")
        wavs = whisper.pad_or_trim(wavs)
        # batch_mel: (batch, 80, 3000)
        batch_mel = whisper.log_mel_spectrogram(wavs, device=self.device)

        with torch.no_grad():
            # (batch, 1500, 1024)
            features = self.whisper_model.embed_audio(batch_mel)

        max_len = int(frame_lens.max().item())
        mask = torch.arange(features.size(1), device=features.device).expand(
            len(frame_lens), -1
        ) < frame_lens.unsqueeze(1)
        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))

        if features.shape[1] >= max_len:
            features = features[:, :max_len, :]
        else:
            padding_frames = max_len - features.shape[1]
            last_frame = features[:, -1:, :]
            padding = last_frame.repeat(1, padding_frames, 1)
            features = torch.cat([features, padding], dim=1)

        features = (features - self.whisper_mean) / self.whisper_std

        return features

    @torch.no_grad()
    def extract_coco_codec(
        self,
        coco_codec_type,
        wav16k,
        wav24k_numpy,
        use_shifted_wav_to_extract_chromagram=False,
        pitch_shift_steps=0,
    ):
        """
        Args:
            coco_codec_type: "style" or "content_style"
            wav16k: [1, T]
            wav24k_numpy: [T]
        Returns:
            codecs: [1, T]. Note that codecs might be not at 50Hz!
        """
        frame_len = len(wav24k_numpy) // self.fmt_cfg.preprocess.hop_size

        if use_shifted_wav_to_extract_chromagram:
            chromagram_feats = self.get_chromagram(
                pitch_shift(wav24k_numpy, sr=24000, n_steps=pitch_shift_steps),
                frame_len,
            )  # [T, 24]
        else:
            chromagram_feats = self.get_chromagram(wav24k_numpy, frame_len)  # [T, 24]

        chromagram_feats = (
            torch.tensor(chromagram_feats, dtype=torch.float)
            .unsqueeze(0)
            .to(self.device)
        )  # [1, T, 24]

        if coco_codec_type == "content_style":
            whisper_feats = self.extract_whisper_features(
                wav16k,
                torch.tensor([frame_len], dtype=torch.long).to(self.device),
            )  # [1, T, D]
            codecs, _ = self.content_style_tokenizer.quantize(
                whisper_feats.to(torch.float32), chromagram_feats.to(torch.float32)
            )
        elif coco_codec_type == "style":
            codecs, _ = self.style_tokenizer.quantize(
                chromagram_feats.to(torch.float32)
            )
        else:
            raise ValueError(f"Unknown coco type: {coco_codec_type}")

        return codecs

    def get_chromagram(self, speech, speech_frames):
        # [24, T] -> [T, 24]
        chromagram = chroma_stft(
            y=speech,
            sr=self.fmt_cfg.preprocess.sample_rate,
            n_fft=self.fmt_cfg.preprocess.n_fft,
            hop_length=self.fmt_cfg.preprocess.hop_size,
            win_length=self.fmt_cfg.preprocess.win_size,
            n_chroma=24,
        ).T

        if chromagram.shape[0] == 0 or speech_frames <= 0:
            raise ValueError("Audio is too short for chromagram extraction")
        if chromagram.shape[0] < speech_frames:
            chromagram = np.pad(
                chromagram, ((0, speech_frames - chromagram.shape[0]), (0, 0)), mode="edge"
            )
        else:
            chromagram = chromagram[:speech_frames]

        return chromagram

    def get_shifted_steps(self, src_wav_path, timbre_ref_wav_path):
        if src_wav_path == timbre_ref_wav_path:
            return 0

        src_f0 = extract_f0_hz(src_wav_path)
        timbre_ref_f0 = extract_f0_hz(timbre_ref_wav_path)

        if src_f0.size == 0 or timbre_ref_f0.size == 0:
            warnings.warn("No F0 frames; skipping pitch shift", RuntimeWarning)
            return 0
        src_f0_median = np.median(src_f0)
        timbre_ref_f0_median = np.median(timbre_ref_f0)
        if (
            not np.isfinite(src_f0_median)
            or not np.isfinite(timbre_ref_f0_median)
            or src_f0_median <= 0
            or timbre_ref_f0_median <= 0
        ):
            warnings.warn("No valid F0; skipping pitch shift", RuntimeWarning)
            return 0

        src_shifted_steps = 12 * np.log2(timbre_ref_f0_median / src_f0_median)
        src_shifted_steps = round(src_shifted_steps)

        if src_shifted_steps > 12:
            src_shifted_steps = src_shifted_steps % 12
        elif src_shifted_steps < -12:
            src_shifted_steps = src_shifted_steps % -12

        return src_shifted_steps

    @torch.no_grad()
    def inference_fm(self, src_wav_path, timbre_ref_wav_path,
                     use_pitch_shift=False, flow_matching_steps=32):
        src_speech, _, src_speech16k = load_wav(src_wav_path, self.device)
        steps = self.get_shifted_steps(src_wav_path, timbre_ref_wav_path) if use_pitch_shift else 0
        src_codecs = self.extract_coco_codec(
            "content_style", src_speech16k, src_speech,
            use_shifted_wav_to_extract_chromagram=use_pitch_shift,
            pitch_shift_steps=steps,
        )
        return self.mel2audio(self.code2mel(src_codecs, timbre_ref_wav_path, flow_matching_steps))

    def get_llm_prompt_prosody(self, prosody_wav_path, style_ref_wav_path,
                               use_pitch_shift, prosody_steps, style_steps):
        prosody_speech, _, prosody_speech16k = load_wav(prosody_wav_path, self.device)
        prosody_ids = self.extract_coco_codec(
            "style", prosody_speech16k, prosody_speech,
            use_shifted_wav_to_extract_chromagram=use_pitch_shift,
            pitch_shift_steps=prosody_steps,
        )
        style_speech, _, style_speech16k = load_wav(style_ref_wav_path, self.device)
        style_ids = self.extract_coco_codec(
            "style", style_speech16k, style_speech,
            use_shifted_wav_to_extract_chromagram=use_pitch_shift,
            pitch_shift_steps=style_steps,
        )
        ids = torch.cat([style_ids, prosody_ids], dim=1)[0].tolist()
        return "<|prosody_start|>" + "".join(f"<|prosody_{i}|>" for i in ids) + "<|prosody_end|>"

    def get_llm_prompt_contentstyle(self, style_ref_wav_path, use_pitch_shift, pitch_shift_steps):
        speech, _, speech16k = load_wav(style_ref_wav_path, self.device)
        ids = self.extract_coco_codec(
            "content_style", speech16k, speech,
            use_shifted_wav_to_extract_chromagram=use_pitch_shift,
            pitch_shift_steps=pitch_shift_steps,
        )[0].tolist()
        return "<|content_style_start|>" + "".join(f"<|content_style_{i}|>" for i in ids)

    def parse_llm_generated_ids(self, generated_ids, llm_input_ids):
        """
        Args:
            generated_ids: [1, T]
            llm_input_ids: [1, T]
        Returns:
            coco_codecs: [1, T]
        """
        input_len = llm_input_ids.shape[1]
        generated_ids = generated_ids[:, input_len:]

        # Eg: <|content_style_start|> <|content_style_1|> <|content_style_2|> <|content_style_end|> <|im_end|>
        generated_text = self.ar_tokenizer.decode(
            generated_ids[0], skip_special_tokens=False
        )

        content_style_ids = [int(i) for i in re.findall(r"<\|content_style_(\d+)\|>", generated_text)]
        if not content_style_ids:
            raise RuntimeError("AR model generated no content-style tokens")
        content_style_ids = (
            torch.tensor(content_style_ids, dtype=torch.long)
            .to(self.device)
            .unsqueeze(0)
        )  # [1, T]

        return content_style_ids

    @torch.no_grad()
    def code2mel(
        self,
        contentstyle_codecs,
        timbre_ref_wav_path,
        flow_matching_steps=32,
    ):
        timbre_ref_speech, timbre_ref_speech24k, timbre_ref_speech16k = load_wav(
            timbre_ref_wav_path,
            self.device,
        )
        timbre_ref_codecs = self.extract_coco_codec(
            "content_style",
            timbre_ref_speech16k,
            timbre_ref_speech,
        )  # [1, T]

        diffusion_input_codecs = torch.cat(
            [timbre_ref_codecs, contentstyle_codecs], dim=1
        )

        # Prepare the condition for diffusion
        diffusion_cond = self.fmt_model.cond_emb(diffusion_input_codecs)  # [1, T, D]
        if self.fmt_model.do_resampling:
            # Align to the frame rate of Mels
            diffusion_cond = self.fmt_model.resampling_layers(
                diffusion_cond.transpose(1, 2)
            ).transpose(1, 2)

        timbre_ref_mels = self.extract_mel_feature(timbre_ref_speech24k)  # [1, T, D]

        # [1, T, D]
        predict_mel_feat = self.fmt_model.reverse_diffusion(
            cond=diffusion_cond,
            prompt=timbre_ref_mels,
            n_timesteps=flow_matching_steps,
        )
        return predict_mel_feat

    @torch.no_grad()
    def mel2audio(self, predict_mel_feat):
        return self.vocoder_model(predict_mel_feat.transpose(1, 2)).detach().cpu()[0]

    @torch.no_grad()
    def inference_ar_and_fm(
        self, target_text, style_ref_wav_path, style_ref_wav_text,
        timbre_ref_wav_path, prosody_wav_path=None, use_prosody_code=True,
        top_k=25, top_p=0.8, temperature=1.0, use_pitch_shift=False,
        flow_matching_steps=32,
    ):
        input_text = f"{style_ref_wav_text} {target_text}" if style_ref_wav_text else target_text
        prompt = gen_chat_prompt(
            input_text, add_assistant_token=True,
            follow_prosody_instruction=use_prosody_code,
        )
        prosody_steps = style_steps = 0
        if use_pitch_shift:
            if prosody_wav_path is not None:
                prosody_steps = self.get_shifted_steps(prosody_wav_path, timbre_ref_wav_path)
            style_steps = self.get_shifted_steps(style_ref_wav_path, timbre_ref_wav_path)
        if use_prosody_code:
            prompt += self.get_llm_prompt_prosody(
                prosody_wav_path, style_ref_wav_path, use_pitch_shift,
                prosody_steps, style_steps,
            )
        prompt += self.get_llm_prompt_contentstyle(
            style_ref_wav_path, use_pitch_shift, style_steps
        )
        input_ids = torch.tensor(
            self.ar_tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long, device=self.device,
        ).unsqueeze(0)
        generated_ids = self.ar_model.generate(
            input_ids=input_ids, min_new_tokens=15, max_new_tokens=500,
            eos_token_id=self.ar_tokenizer.eos_token_id, do_sample=True,
            top_k=top_k, top_p=top_p, temperature=temperature,
        )
        codecs = self.parse_llm_generated_ids(generated_ids, input_ids)
        return self.mel2audio(self.code2mel(codecs, timbre_ref_wav_path, flow_matching_steps))
