"""AR + FM CLI compatibility entry point."""
from models.svc.vevo2.cli import main

if __name__ == "__main__":
    main(tasks=("tts", "edit", "style", "melody"))
