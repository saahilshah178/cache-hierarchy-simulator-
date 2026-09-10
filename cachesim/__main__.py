"""Allow ``python -m cachesim``; exits with the CLI's status code."""

import sys

from cachesim.cli import main

if __name__ == "__main__":
    sys.exit(main())
