"""Text and melody conditioned synthesis CLI."""
from .cli import main

if __name__ == "__main__":
    main(tasks=("tts", "edit", "style", "melody"))
