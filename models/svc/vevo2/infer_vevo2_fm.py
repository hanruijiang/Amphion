"""FM-only CLI compatibility entry point."""
import sys
from models.svc.vevo2.cli import main

if __name__ == "__main__":
    main(["convert", *sys.argv[1:]], tasks=("convert",))
