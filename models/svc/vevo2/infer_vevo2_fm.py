"""Voice and singing conversion CLI."""
import sys
from .cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] != "convert":
        argv = ["convert", *argv]
    main(argv, tasks=("convert",))
