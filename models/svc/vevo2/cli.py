"""Command-line entry point for Vevo2 inference (no model imports until needed)."""
import argparse
from pathlib import Path

TASKS = ("convert", "tts", "edit", "style", "melody")


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
                       help="Use cached model weights only")
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
    from .audio import save_audio, validate_audio_inputs
    from .checkpoint import load_pipeline

    validate_audio_inputs(args)
    pipeline = load_pipeline(args)
    audio = run_task(pipeline, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_audio(audio, output_path=str(args.output))
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
