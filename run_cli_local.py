#!/usr/bin/env python3
"""Local runner for the Click CLI from within this folder.

Usage (from this directory):
  python run_cli_local.py --query "your prompt" --hyde --num 5
"""
import os
import sys


def main():
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(pkg_dir, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from comp_l0_queryrewrite.cli.rewrite_cli import main as cli_main

    # Delegate to Click command
    cli_main(standalone_mode=False)


if __name__ == "__main__":
    # Allow Click to parse argv passed to this script
    try:
        main()
    except SystemExit as e:
        # Click may call sys.exit; propagate exit code cleanly
        raise
