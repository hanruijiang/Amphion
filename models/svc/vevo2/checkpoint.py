"""Model resources and weight loading."""
import os
from pathlib import Path

import torch
import whisper
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

MODEL_REPO = "RMSnow/Vevo2"
MODEL_REVISION = "2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8"
FM_DIR = "acoustic_modeling/fm_emilia101k_singnet7k_repa"
AR_DIR = "contentstyle_modeling/posttrained"
CONTENT_DIR = "tokenizer/contentstyle_fvq16384_12.5hz"
PROSODY_DIR = "tokenizer/prosody_fvq512_6.25hz"
AR_FILES = (
    "amphion_config.json", "config.json", "generation_config.json",
    "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
)
TOKENIZER_TRAINING_PREFIXES = (
    "upsample_layers.", "decoder.", "whisper_output_layer.",
    "chromagram_output_layer.",
)
FM_TRAINING_PREFIXES = ("repa_mlp_layer.",)


def load_weights(model, path, device, ignored_prefixes=()):
    state_dict = load_file(str(path), device="cpu")
    state_dict = {
        name: value for name, value in state_dict.items()
        if not name.startswith(ignored_prefixes)
    }
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def load_whisper(device, local_files_only):
    if local_files_only:
        cache = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
        path = cache / "medium.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Whisper medium checkpoint not found: {path}")
        return whisper.load_model(str(path), device=device).eval()
    return whisper.load_model("medium", device=device).eval()


def load_whisper_stats(path, device):
    stats = torch.load(path, map_location="cpu", weights_only=True)
    mean, std = stats["mean"], stats["std"]
    if mean.shape != (1024,) or std.shape != (1024,):
        raise ValueError(f"Invalid Whisper statistics: {path}")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError(f"Invalid Whisper statistics: {path}")
    return mean.to(device), std.to(device)


def load_pipeline(args):
    from .pipeline import Vevo2InferencePipeline

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(device)
    use_ar = args.task != "convert"
    use_prosody = args.task in ("edit", "style", "melody")
    paths = [
        CONTENT_DIR + "/model.safetensors",
        FM_DIR + "/config.json",
        FM_DIR + "/model.safetensors",
        FM_DIR + "/whisper_stats.pt",
        "vocoder/config.json",
        "vocoder/model.safetensors",
    ]
    if use_ar:
        paths += [f"{AR_DIR}/{name}" for name in AR_FILES]
    if use_prosody:
        paths.append(PROSODY_DIR + "/model.safetensors")
    local = Path(snapshot_download(
        repo_id=MODEL_REPO, revision=MODEL_REVISION,
        local_dir=str(args.ckpt_dir), local_files_only=args.local_files_only,
        allow_patterns=paths,
    )).resolve()
    return Vevo2InferencePipeline(local, device, use_ar, use_prosody, args.local_files_only)
