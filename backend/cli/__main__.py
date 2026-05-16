"""Argparse dispatcher for `python -m cli ...` / the `alphatest` shell wrapper.

Each subcommand module registers itself by calling `add_subparser` here.
Keeping the dispatcher tiny means there's nothing for tests to mock — the
real entry point and the test-driven entry point are the same code path.
"""

from __future__ import annotations

import argparse
import sys

from cli import list_alphas, run, shuffle, verify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alphatest",
        description="QuantLab CLI — backtest, shuffle-test, list, and verify alphas.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run.add_subparser(sub)
    shuffle.add_subparser(sub)
    list_alphas.add_subparser(sub)
    verify.add_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
