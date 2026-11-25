# Vevo2 CLI inference

A minimal inference-only fork of [Amphion](https://github.com/open-mmlab/Amphion/tree/main/models/svc/vevo2). No training, evaluation, web UI or other model recipes. The codec, Vocos vocoder, flow-matching transformer and Llama NAR implementation under `models/` are runtime dependencies of Vevo2.

## Installation

Use Python 3.11 (the reference Space's version), FFmpeg on PATH and a CUDA-capable GPU for practical inference. Install matching PyTorch wheels for your platform if needed (see https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/hanruijiang/Amphion.git
cd Amphion
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m models.svc.vevo2 --help
```

## Inference

Use your own audio files. Output is mono 24 kHz WAV. The first run downloads the required weights from [RMSnow/Vevo2](https://huggingface.co/RMSnow/Vevo2) at the pinned revision and Whisper medium weights; allow several GB of storage. FM-only conversion does not download the AR model or prosody tokenizer. The other tasks load AR + FM.

```bash
# Voice / singing conversion (FM)
python -m models.svc.vevo2 convert --input source.wav --reference voice.wav --output output/convert.wav

# Zero-shot TTS / text-to-singing
python -m models.svc.vevo2 tts --input style.wav --reference-text "Reference transcript" --text "Hello world" --output output/tts.wav

# Speech / singing editing
python -m models.svc.vevo2 edit --input source.wav --reference-text "Original lyrics" --text "New lyrics" --output output/edit.wav

# Singing style conversion (keep source timbre and melody)
python -m models.svc.vevo2 style --input source.wav --reference style.wav --reference-text "Style reference lyrics" --text "Original source lyrics" --output output/style.wav

# Humming / instrument to singing
python -m models.svc.vevo2 melody --input melody.wav --reference voice.wav --reference-text "Reference lyrics" --text "Target lyrics" --output output/melody.wav
```

Each task supports `--device auto|cuda|cpu`, `--steps 32`, `--ckpt-dir ckpts/Vevo2`, `--local-files-only`, and `--no-pitch-shift`. Use `--help` after the task for its options. CPU inference may be very slow. `--local-files-only` applies to Vevo2 weights; cache Whisper medium separately before running without a network. For TTS and melody, `--timbre-reference` optionally specifies a separate voice. Provide accurate reference transcripts when available.

The original module names remain as CLI wrappers:
```bash
python -m models.svc.vevo2.infer_vevo2_fm --help
python -m models.svc.vevo2.infer_vevo2_ar --help
```

## Tests

```bash
python -m unittest discover -s tests
python -m compileall -q models
```

Tests check CLI routing and arguments without downloading checkpoints. Real audio generation requires the model weights and compatible hardware.

## Sources and license

- Upstream [Amphion](https://github.com/open-mmlab/Amphion) code: MIT (see LICENSE).
- [Vevo2 paper](https://arxiv.org/abs/2508.16332) — please cite the original authors.
- [Vevo2 model weights](https://huggingface.co/RMSnow/Vevo2) have a separate model license; consult the model card before use (CC BY-NC-ND 4.0).
- [Reference Space](https://huggingface.co/spaces/teamup-tech/Vevo2) informed runtime dependencies and checkpoint handling.
