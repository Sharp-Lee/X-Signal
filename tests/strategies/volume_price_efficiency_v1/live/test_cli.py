import pytest

from xsignal.strategies.volume_price_efficiency_v1.live import status_cli
from xsignal.strategies.volume_price_efficiency_v1.live.cli import (
    build_parser,
    run_status_command,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


def test_cli_has_replay_status_and_reconcile_commands():
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    assert {"replay", "status", "reconcile"} <= set(subcommands)


def test_status_requires_database_path():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["status"])


def test_status_accepts_json_and_no_system_flags():
    parser = build_parser()

    args = parser.parse_args(["status", "--db", "live.sqlite", "--json", "--no-system"])

    assert args.db.name == "live.sqlite"
    assert args.json is True
    assert args.no_system is True


def test_run_status_command_prints_text_snapshot_without_system_checks(tmp_path, capsys):
    db = tmp_path / "live.sqlite"
    store = LiveStore.open(db)
    store.initialize()

    exit_code = run_status_command(db=db, json_output=False, collect_system=False)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OVERALL OK" in captured.out
    assert "SERVICE active=unknown" in captured.out


def test_run_status_command_returns_nonzero_for_warning_snapshot(tmp_path, capsys):
    db = tmp_path / "live.sqlite"
    store = LiveStore.open(db)
    store.initialize()

    exit_code = run_status_command(
        db=db,
        json_output=True,
        collect_system=False,
        system_snapshot={
            "system_available": True,
            "service_active": False,
            "live_service_active": False,
            "live_guard_present": False,
            "revision": "abc123",
            "sockets": [],
            "journal": {"reconcile_clean": 0, "stream_errors": 0, "rest_429": 0},
        },
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert '"overall": "WARN"' in captured.out
    assert "testnet_service_inactive" in captured.out


def test_standalone_status_cli_delegates_to_status_command(monkeypatch):
    calls = []

    def fake_run_status_command(**kwargs):
        calls.append(kwargs)
        return 7

    monkeypatch.setattr(status_cli, "run_status_command", fake_run_status_command)

    exit_code = status_cli.main(["--db", "live.sqlite", "--json", "--no-system"])

    assert exit_code == 7
    assert calls == [
        {
            "db": status_cli.Path("live.sqlite"),
            "json_output": True,
            "collect_system": False,
        }
    ]
