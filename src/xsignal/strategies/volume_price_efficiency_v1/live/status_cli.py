from __future__ import annotations

import argparse
from pathlib import Path

from xsignal.strategies.volume_price_efficiency_v1.live.cli import run_status_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xsignal-vpe-status")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-system", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_status_command(
        db=args.db,
        json_output=args.json,
        collect_system=not args.no_system,
    )
