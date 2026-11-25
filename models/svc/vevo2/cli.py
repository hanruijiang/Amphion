"""Command-line entry point for Vevo2 inference (no model imports until needed)."""
import argparse
from pathlib import Path

MODEL_REPO = "RMSnow/Vevo2"
MODEL_REVISION = "2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8"
FM_DIR = "acoustic_modeling/fm_emilia101k_singnet7k_repa"
AR_DIR = "contentstyle_modeling/posttrained"
CONTENT_DIR = "tokenizer/contentstyle_fvq16384_12.5hz"
PROSODY_DIR = "tokenizer/prosody_fvq512_6.25hz"
TASKS = ("convert", "tts", "edit", "style", "melody")

def load_pipeline(args):
    import torch
    from huggingface_hub import snapshot_download
    from models.svc.vevo2.vevo2_utils import Vevo2InferencePipeline

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    paths = [
        CONTENT_DIR + "/model.safetensors",
        FM_DIR + "/config.json",
        FM_DIR + "/model.safetensors",
        FM_DIR + "/whisper_stats.pt",
        "vocoder/config.json",
        "vocoder/model.safetensors",
    ]
    use_ar = args.task != "convert"
    use_prosody = args.task in ("edit", "style", "melody")
    if use_ar:
        paths += [AR_DIR + "/*"]
    if use_prosody:
        paths += [PROSODY_DIR + "/model.safetensors"]
    local = Path(snapshot_download(
        repo_id=MODEL_REPO, revision=MODEL_REVISION,
        local_dir=str(args.ckpt_dir), local_files_only=args.local_files_only,
        allow_patterns=paths,
        ignore_patterns=["*/optimizer.*", "*/scheduler.*", "*/rng_state*",
                         "*/random_states_*", "*/trainer_state.json",
                         "*/training_args.bin"],
    )).resolve()
    return Vevo2InferencePipeline(
        content_style_tokenizer_ckpt_path=str(local / CONTENT_DIR),
        prosody_tokenizer_ckpt_path=str(local / PROSODY_DIR) if use_prosody else None,
        ar_cfg_path=str(local / AR_DIR / "amphion_config.json") if use_ar else None,
        ar_ckpt_path=str(local / AR_DIR) if use_ar else None,
        fmt_cfg_path=str(local / FM_DIR / "config.json"),
        fmt_ckpt_path=str(local / FM_DIR / "model.safetensors"),
        vocoder_cfg_path=str(local / "vocoder/config.json"),
        vocoder_ckpt_path=str(local / "vocoder/model.safetensors"),
        device=torch.device(device),
    )

def build_parser(tasks=TASKS):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    descriptions = {
        "convert": "Style-preserving voice / singing conversion (FM only)",
        "tts": "Text-to-speech / text-to-singing (AR + FM)",
        "edit": "Speech / singing editing (AR + FM)",
        "style": "Singing style conversion (AR + FM)",
        "melody": "Humming / instrument to singing (AR + FM)",
    }
    for task in tasks:
        p = sub.add_parser(task, help=descriptions[task], description=descriptions[task])
        p.add_argument("--input", type=Path, required=True,
                       help="Source audio, or style reference for tts, or melody for melody")
        p.add_argument("--output", type=Path, required=True, help="Output WAV path")
        p.add_argument("--ckpt-dir", type=Path, default=Path("ckpts/Vevo2"))
        p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
        p.add_argument("--steps", type=int, default=32, help="Flow-matching steps")
        p.add_argument("--local-files-only", action="store_true",
                       help="Do not download Vevo2 weights; Whisper must already be cached")
        p.add_argument("--pitch-shift", action=argparse.BooleanOptionalAction,
                       default=task in ("convert", "style", "melody"))
        if task in ("convert", "style", "melody", "tts"):
            p.add_argument("--reference", type=Path, required=task in ("convert", "style", "melody"),
                           help="Reference voice/style; optional timbre reference for tts")
        if task in ("tts", "melody"):
            p.add_argument("--timbre-reference", type=Path, help="Separate timbre reference")
        if task != "convert":
            p.add_argument("--text", required=True, help="Target text or lyrics (original transcript for style)")
            p.add_argument("--reference-text", default="", help="Style reference transcript, if available")
    return parser

def validate(parser, args):
    if args.steps < 1:
        parser.error("--steps must be positive")
    for name in ("input", "reference", "timbre_reference"):
        path = getattr(args, name, None)
        if path is not None and not path.is_file():
            parser.error(f"{name}: file not found: {path}")
        if path is not None and path.resolve() == args.output.resolve():
            parser.error("--output must not overwrite an input file")
    if args.output.suffix.lower() != ".wav":
        parser.error("--output must end in .wav")
    if args.task != "convert" and not args.text.strip():
        parser.error("--text must not be empty")

def run_task(pipe, args):
    source = str(args.input)
    reference = str(args.reference) if getattr(args, "reference", None) else None
    options = dict(use_pitch_shift=args.pitch_shift, flow_matching_steps=args.steps)
    if args.task == "convert":
        return pipe.inference_fm(src_wav_path=source, timbre_ref_wav_path=reference, **options)
    if args.task == "tts":
        style, timbre, prosody = source, reference or source, None
    elif args.task == "edit":
        style = timbre = prosody = source
    elif args.task == "style":
        style, timbre, prosody = reference, source, source
    else:
        style, timbre, prosody = reference, reference, source
    if getattr(args, "timbre_reference", None):
        timbre = str(args.timbre_reference)
    return pipe.inference_ar_and_fm(
        target_text=args.text, style_ref_wav_path=style,
        style_ref_wav_text=args.reference_text, timbre_ref_wav_path=timbre,
        prosody_wav_path=prosody, use_prosody_code=prosody is not None, **options,
    )

def main(argv=None, tasks=TASKS):
    parser = build_parser(tasks)
    args = parser.parse_args(argv)
    validate(parser, args)
    from models.svc.vevo2.vevo2_utils import save_audio
    pipeline = load_pipeline(args)
    audio = run_task(pipeline, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_audio(audio, output_path=str(args.output))
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
