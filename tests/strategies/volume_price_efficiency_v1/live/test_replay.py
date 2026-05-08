from xsignal.strategies.volume_price_efficiency_v1.live.replay import ReplaySummary


def test_replay_summary_counts_are_plain_dataclass():
    summary = ReplaySummary(processed_bars=10, accepted_signals=2, submitted_orders=4)
    assert summary.processed_bars == 10
    assert summary.accepted_signals == 2
    assert summary.submitted_orders == 4
