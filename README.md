# Vevo2 CLI inference

Command-line speech and singing inference with Vevo2.

## Installation

Use Python 3.11, FFmpeg and a CUDA-capable GPU.

```bash
git clone --branch vevo2-cli-only --single-branch https://github.com/hanruijiang/Amphion.git
cd Amphion
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m models.svc.vevo2 --help
```

## Inference

```bash
# Voice / singing conversion
python -m models.svc.vevo2 convert --input source.wav --reference voice.wav --output output/convert.wav

# Text-to-speech / text-to-singing
python -m models.svc.vevo2 tts --input style.wav --reference-text "Reference transcript" --text "Hello world" --output output/tts.wav

# Speech / singing editing
python -m models.svc.vevo2 edit --input source.wav --reference-text "Original lyrics" --text "New lyrics" --output output/edit.wav

# Singing style conversion
python -m models.svc.vevo2 style --input source.wav --reference style.wav --reference-text "Style reference lyrics" --text "Original source lyrics" --output output/style.wav

# Humming / instrument to singing
python -m models.svc.vevo2 melody --input melody.wav --reference voice.wav --reference-text "Reference lyrics" --text "Target lyrics" --output output/melody.wav
```

Each task supports `--device auto|cuda|cpu`, `--steps 32`, `--ckpt-dir ckpts/Vevo2`, `--local-files-only` and `--no-pitch-shift`. TTS and melody also support `--timbre-reference`. Use the task's `--help` for all options.

Models are downloaded from [RMSnow/Vevo2](https://huggingface.co/RMSnow/Vevo2) and Whisper medium on the first run. For offline inference, cache these weights first and use `--local-files-only`. Audio inputs must be at least 0.1 seconds; audio used for Whisper features must be at most 30 seconds. Outputs are mono 24 kHz WAV.

Task-specific entry points:

```bash
python -m models.svc.vevo2.infer_vevo2_fm --help
python -m models.svc.vevo2.infer_vevo2_ar --help
```

## Tests

```bash
python -m unittest discover -s tests
python -m compileall -q models
```

## Sources and license

- [Amphion](https://github.com/open-mmlab/Amphion): MIT (see LICENSE).
- [Vevo2 paper](https://arxiv.org/abs/2508.16332).
- [Vevo2 model weights](https://huggingface.co/RMSnow/Vevo2): CC BY-NC-ND 4.0.
