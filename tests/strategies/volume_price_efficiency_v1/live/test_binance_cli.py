from xsignal.strategies.volume_price_efficiency_v1.live import cli


def test_cli_has_testnet_smoke_command():
    parser = cli.build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    args = parser.parse_args(["testnet-smoke", "--symbol", "BTCUSDT"])
    assert "testnet-smoke" in subcommands
    assert args.command == "testnet-smoke"
    assert args.symbol == "BTCUSDT"
    assert not args.submit_test_order


def test_testnet_smoke_missing_env_returns_nonzero(monkeypatch, capsys):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_SECRET_KEY", raising=False)

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
