from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

from xsignal.strategies.volume_price_efficiency_v1.live import cli
from xsignal.strategies.volume_price_efficiency_v1.live.models import SymbolMetadata


def test_cli_has_testnet_smoke_command():
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    args = parser.parse_args(["testnet-smoke", "--symbol", "BTCUSDT"])
    assert "testnet-smoke" in subcommands
    assert args.command == "testnet-smoke"
    assert args.symbol == "BTCUSDT"
    assert not args.submit_test_order


def test_testnet_smoke_missing_env_returns_nonzero(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_SECRET_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    result = cli.main(["testnet-smoke", "--symbol", "BTCUSDT"])

    captured = capsys.readouterr()
    assert result == 2
    assert "BINANCE_API_KEY" in captured.err


def test_testnet_smoke_runs_read_only_with_injected_client(capsys):
    class FakeBroker:
        def get_position_mode(self):
            return "one_way"

        def get_multi_assets_mode(self):
            return "single_asset_usdt"

    class FakeRestClient:
        def __init__(self) -> None:
            self.calls = []

        def request(self, method, path, *, signed=False, params=None):
            self.calls.append((method, path, signed, params or {}))
            if path == "/fapi/v1/time":
                return {"serverTime": 123}
            if path == "/fapi/v1/exchangeInfo":
                return {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}]}
            if path == "/fapi/v3/account":
                return {
                    "totalMarginBalance": "100",
                    "availableBalance": "90",
                    "totalInitialMargin": "0",
                }
            return {}

    rest_client = FakeRestClient()

    result = cli.run_testnet_smoke(
        symbol="BTCUSDT",
        submit_test_order=False,
        quantity=0.001,
        rest_client=rest_client,
        broker=FakeBroker(),
    )

    captured = capsys.readouterr()
    assert result == 0
    assert '"symbol": "BTCUSDT"' in captured.out
    assert not any(call[1] == "/fapi/v1/order/test" for call in rest_client.calls)


def test_testnet_smoke_optional_test_order_uses_order_test_endpoint(capsys):
    class FakeBroker:
        def get_position_mode(self):
            return "one_way"

        def get_multi_assets_mode(self):
            return "single_asset_usdt"

        def test_order(self, *, symbol, side, order_type, quantity, client_order_id):
            self.test_order_args = (symbol, side, order_type, quantity, client_order_id)
            return {}

    class FakeRestClient:
        def request(self, method, path, *, signed=False, params=None):
            if path == "/fapi/v1/time":
                return {"serverTime": 123}
            if path == "/fapi/v1/exchangeInfo":
                return {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}]}
            if path == "/fapi/v3/account":
                return {
                    "totalMarginBalance": "100",
                    "availableBalance": "90",
                    "totalInitialMargin": "0",
                }
            return {}

    broker = FakeBroker()

    result = cli.run_testnet_smoke(
        symbol="BTCUSDT",
        submit_test_order=True,
        quantity=0.001,
        rest_client=FakeRestClient(),
        broker=broker,
    )

    assert result == 0
    assert broker.test_order_args[0:4] == ("BTCUSDT", "BUY", "MARKET", 0.001)
    assert broker.test_order_args[4].startswith("XV1T")


def test_cli_has_guarded_testnet_lifecycle_command():
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    args = parser.parse_args(
        [
            "testnet-lifecycle",
            "--symbol",
            "BTCUSDT",
            "--quantity",
            "0.001",
            "--stop-offset-pct",
            "0.05",
        ]
    )

    assert "testnet-lifecycle" in subcommands
    assert args.command == "testnet-lifecycle"
    assert args.symbol == "BTCUSDT"
    assert args.quantity == 0.001
    assert args.stop_offset_pct == 0.05
    assert not args.i_understand_testnet_order


def test_cli_has_guarded_testnet_reconcile_command(tmp_path):
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    args = parser.parse_args(
        [
            "testnet-reconcile",
            "--db",
            str(tmp_path / "live.sqlite"),
            "--symbol",
            "BTCUSDT",
            "--repair",
        ]
    )

    assert "testnet-reconcile" in subcommands
    assert args.command == "testnet-reconcile"
    assert args.db == tmp_path / "live.sqlite"
    assert args.symbol == ["BTCUSDT"]
    assert args.repair
    assert not args.i_understand_testnet_order


def test_cli_has_run_cycle_and_live_smoke_commands(tmp_path):
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    args = parser.parse_args(
        [
            "run-cycle",
            "--mode",
            "live",
            "--db",
            str(tmp_path / "live.sqlite"),
            "--symbol",
            "BTCUSDT",
            "--max-symbols",
            "3",
            "--i-understand-live-order",
        ]
    )
    smoke_args = parser.parse_args(["live-smoke", "--symbol", "BTCUSDT"])

    assert "run-cycle" in subcommands
    assert "live-smoke" in subcommands
    assert args.command == "run-cycle"
    assert args.mode == "live"
    assert args.symbol == ["BTCUSDT"]
    assert args.max_symbols == 3
    assert args.i_understand_live_order
    assert smoke_args.command == "live-smoke"


def test_cli_has_stream_daemon_command(tmp_path):
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    args = parser.parse_args(
        [
            "stream-daemon",
            "--mode",
            "testnet",
            "--db",
            str(tmp_path / "stream.sqlite"),
            "--interval",
            "1h",
            "--interval",
            "4h",
            "--max-symbols",
            "10",
            "--max-streams",
            "200",
            "--recovery-sleep-ms",
            "150",
            "--stop-after-events",
            "3",
        ]
    )

    assert "stream-daemon" in subcommands
    assert args.command == "stream-daemon"
    assert args.mode == "testnet"
    assert args.db == tmp_path / "stream.sqlite"
    assert args.interval == ["1h", "4h"]
    assert args.max_symbols == 10
    assert args.max_streams == 200
    assert args.recovery_sleep_ms == 150
    assert args.stop_after_events == 3


def test_testnet_lifecycle_requires_explicit_acknowledgement(capsys):
    result = cli.run_testnet_lifecycle_command(
        symbol="BTCUSDT",
        quantity=0.001,
        stop_offset_pct=0.05,
        acknowledge=False,
        broker=object(),
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "--i-understand-testnet-order" in captured.err


def test_credentials_load_local_secret_file_without_overriding_env(monkeypatch, tmp_path):
    env_file = tmp_path / "binance-testnet.env"
    env_file.write_text(
        "BINANCE_API_KEY=file-api\nBINANCE_SECRET_KEY=file-secret\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_SECRET_KEY", raising=False)

    credentials = cli._credentials_from_env(env_file=env_file)

    assert credentials.api_key == "file-api"
    assert credentials.secret_key == "file-secret"

    monkeypatch.setenv("BINANCE_API_KEY", "env-api")
    monkeypatch.setenv("BINANCE_SECRET_KEY", "env-secret")

    credentials = cli._credentials_from_env(env_file=env_file)

    assert credentials.api_key == "env-api"
    assert credentials.secret_key == "env-secret"


def test_testnet_lifecycle_runs_runner_and_prints_result(capsys):
    class FakeBroker:
        def get_symbol_metadata(self, symbol):
            return SymbolMetadata(
                symbol=symbol,
                status="TRADING",
                min_notional=5.0,
                quantity_step=0.001,
                price_tick=0.1,
                supports_stop_market=True,
                trigger_protect=0.05,
                updated_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
            )

    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            symbol="BTCUSDT",
            quantity=0.001,
            stop_offset_pct=0.05,
            stop_price=95000.0,
            entry_client_order_id="XV1TE...",
            stop_client_order_id="XV1TS...",
            close_client_order_id="XV1TM...",
            opened_position_amount=0.001,
            final_position_amount=0.0,
        )

    result = cli.run_testnet_lifecycle_command(
        symbol="BTCUSDT",
        quantity=0.001,
        stop_offset_pct=0.05,
        acknowledge=True,
        broker=FakeBroker(),
        lifecycle_runner=fake_runner,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert calls[0]["price_tick"] == 0.1
    assert calls[0]["symbol_rules"].market_quantity_step == Decimal("0.00100000")
    assert '"final_position_amount": 0.0' in captured.out
    assert "secret" not in captured.out.lower()


def test_testnet_lifecycle_passes_store_when_db_is_provided(tmp_path, capsys):
    class FakeBroker:
        def get_symbol_metadata(self, symbol):
            return SymbolMetadata(
                symbol=symbol,
                status="TRADING",
                min_notional=5.0,
                quantity_step=0.001,
                price_tick=0.1,
                supports_stop_market=True,
                trigger_protect=0.05,
                updated_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
            )

    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        assert kwargs["store"].get_position_state("missing") is None
        return SimpleNamespace(
            symbol="BTCUSDT",
            quantity=0.001,
            stop_offset_pct=0.05,
            stop_price=95000.0,
            entry_client_order_id="XV1TE...",
            stop_client_order_id="XV1TS...",
            close_client_order_id="XV1TM...",
            opened_position_amount=0.001,
            final_position_amount=0.0,
        )

    result = cli.run_testnet_lifecycle_command(
        symbol="BTCUSDT",
        quantity=0.001,
        stop_offset_pct=0.05,
        acknowledge=True,
        db=tmp_path / "live.sqlite",
        broker=FakeBroker(),
        lifecycle_runner=fake_runner,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert calls[0]["store"] is not None
    assert '"symbol": "BTCUSDT"' in captured.out


def test_testnet_reconcile_repair_requires_explicit_acknowledgement(tmp_path, capsys):
    result = cli.run_testnet_reconcile_command(
        db=tmp_path / "live.sqlite",
        symbols=["BTCUSDT"],
        repair=True,
        acknowledge=False,
        broker=object(),
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "--i-understand-testnet-order" in captured.err


def test_testnet_reconcile_runs_runner_and_prints_summary(tmp_path, capsys):
    class Summary:
        error_count = 0

        def to_dict(self):
            return {
                "environment": "testnet",
                "allow_repair": False,
                "error_count": 0,
                "findings": [{"symbol": "BTCUSDT", "status": "CLEAN"}],
            }

    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return Summary()

    result = cli.run_testnet_reconcile_command(
        db=tmp_path / "live.sqlite",
        symbols=["BTCUSDT"],
        repair=False,
        acknowledge=False,
        broker=object(),
        reconcile_runner=fake_runner,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert calls[0]["environment"] == "testnet"
    assert not calls[0]["allow_repair"]
    assert '"status": "CLEAN"' in captured.out
    assert "secret" not in captured.out.lower()


def test_run_cycle_refuses_live_without_ack_and_enable_file(tmp_path, capsys):
    result = cli.run_live_cycle_command(
        mode="live",
        db=tmp_path / "live.sqlite",
        symbols=["BTCUSDT"],
        max_symbols=1,
        lookback_bars=120,
        env_file=None,
        acknowledge_live=False,
        live_enabled=False,
        broker=object(),
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "live trading requires" in captured.err


def test_stream_daemon_refuses_live_without_ack_and_enable_file(tmp_path, capsys):
    result = cli.run_stream_daemon_command(
        mode="live",
        db=tmp_path / "stream.sqlite",
        intervals=["1h"],
        max_symbols=1,
        lookback_bars=120,
        env_file=None,
        acknowledge_live=False,
        live_enabled=False,
        daemon_runner=lambda **kwargs: 0,
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "live trading requires" in captured.err


def test_stream_daemon_uses_injected_runner(tmp_path):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return 0

    result = cli.run_stream_daemon_command(
        mode="testnet",
        db=tmp_path / "stream.sqlite",
        intervals=["1h", "4h"],
        max_symbols=2,
        lookback_bars=120,
        env_file=None,
        acknowledge_live=False,
        live_enabled=False,
        max_streams=200,
        credentials=object(),
        daemon_runner=fake_runner,
    )

    assert result == 0
    assert calls[0]["config"].mode == "testnet"
    assert calls[0]["config"].db_path == tmp_path / "stream.sqlite"
    assert calls[0]["config"].intervals == ("1h", "4h")
    assert calls[0]["config"].max_streams == 200
    assert calls[0]["credentials"] is not None


def test_run_cycle_uses_injected_dependencies_and_prints_result(tmp_path, capsys):
    class Summary:
        error_count = 0

    class Result:
        def to_dict(self):
            return {"entries": 1, "scanned_symbols": 1}

    calls = []

    def fake_cycle_runner(**kwargs):
        calls.append(kwargs)
        return Result()

    result = cli.run_live_cycle_command(
        mode="testnet",
        db=tmp_path / "live.sqlite",
        symbols=["BTCUSDT"],
        max_symbols=1,
        lookback_bars=120,
        env_file=None,
        acknowledge_live=False,
        live_enabled=False,
        broker=object(),
        arrays=object(),
        account=object(),
        metadata_by_symbol={"BTCUSDT": object()},
        prices_by_symbol={"BTCUSDT": 100.0},
        cycle_runner=fake_cycle_runner,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert calls[0]["environment"] == "testnet"
    assert calls[0]["now"].tzinfo is not None
    assert '"entries": 1' in captured.out
