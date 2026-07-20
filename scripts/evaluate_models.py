"""Evaluate champion — delegates to retention evaluate."""
import sys


def main():
    sys.argv = ["retention", "evaluate"]
    from retention.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
