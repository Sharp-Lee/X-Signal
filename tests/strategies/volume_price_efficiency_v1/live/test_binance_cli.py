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
