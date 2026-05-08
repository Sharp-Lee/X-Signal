import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.cli import build_parser


def test_cli_has_replay_status_and_reconcile_commands():
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    assert {"replay", "status", "reconcile"} <= set(subcommands)


def test_status_requires_database_path():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["status"])
