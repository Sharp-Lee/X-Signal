from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xsignal-vpe-live")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--root", type=Path, default=Path("data"))
    replay.add_argument("--db", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--db", type=Path, required=True)

    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("--db", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
