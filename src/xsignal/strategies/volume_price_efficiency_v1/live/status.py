from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess


DEFAULT_TESTNET_SERVICE = "xsignal-vpe-testnet-stream-daemon.service"
DEFAULT_LIVE_SERVICE = "xsignal-vpe-live-stream-daemon.service"
DEFAULT_REVISION_FILE = Path("/opt/x-signal/DEPLOY_REVISION")
DEFAULT_LIVE_GUARD_FILE = Path("/etc/xsignal/enable-live-trading")


def build_status_snapshot(
    *,
    db_path: Path,
    now: datetime | None = None,
    system_snapshot: dict[str, object] | None = None,
) -> dict[str, object]:
    now = now or datetime.now(timezone.utc)
    system_snapshot = system_snapshot or {}
    system_available = bool(system_snapshot.get("system_available", True))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    market_bars = _market_bar_summary(conn)
    cursors = _cursor_summary(conn, now=now)
    positions = _position_summary(conn)
    orders = _order_summary(conn)
    sockets = list(system_snapshot.get("sockets", []))
    journal = dict(system_snapshot.get("journal", {}))
    warnings = _warnings(
        system_available=system_available,
        service_active=bool(system_snapshot.get("service_active", False)),
        live_service_active=bool(system_snapshot.get("live_service_active", False)),
        live_guard_present=bool(system_snapshot.get("live_guard_present", False)),
        sockets=sockets,
        journal=journal,
        positions=positions,
        orders=orders,
        cursors=cursors,
    )
    return {
        "overall": "WARN" if warnings else "OK",
        "warnings": warnings,
        "captured_at": now.isoformat(),
        "revision": system_snapshot.get("revision"),
        "system_available": system_available,
        "service_active": bool(system_snapshot.get("service_active", False)),
        "live_service_active": bool(system_snapshot.get("live_service_active", False)),
        "live_guard_present": bool(system_snapshot.get("live_guard_present", False)),
        "sockets": sockets,
        "journal": journal,
        "market_bars": market_bars,
        "cursors": cursors,
        "positions": positions,
        "orders": orders,
    }


def collect_system_snapshot(
    *,
    service: str = DEFAULT_TESTNET_SERVICE,
    live_service: str = DEFAULT_LIVE_SERVICE,
    revision_file: Path = DEFAULT_REVISION_FILE,
    live_guard_file: Path = DEFAULT_LIVE_GUARD_FILE,
    journal_since: str = "30 minutes ago",
    runner=subprocess.run,
) -> dict[str, object]:
    try:
        service_active = _systemctl_is_active(service, runner=runner)
        live_service_active = _systemctl_is_active(live_service, runner=runner)
        pid = _systemctl_main_pid(service, runner=runner)
    except OSError:
        return _local_system_unavailable_snapshot(
            revision_file=revision_file,
            live_guard_file=live_guard_file,
        )
    sockets = []
    if pid and pid != "0":
        sockets = parse_socket_rows(
            _run_text(
                ["bash", "-lc", f'timeout 5 ss -ntp | grep "{pid}" || true'],
                runner=runner,
            )
        )
    journal_text = _run_text(
        ["journalctl", "-u", service, "--since", journal_since, "--no-pager"],
        runner=runner,
    )
    return {
        "service_active": service_active,
        "system_available": True,
        "live_service_active": live_service_active,
        "live_guard_present": live_guard_file.exists(),
        "revision": _read_text(revision_file),
        "main_pid": pid,
        "sockets": sockets,
        "journal": parse_journal_summary(journal_text),
    }


def parse_socket_rows(output: str) -> list[dict[str, object]]:
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            continue
        if not parts[1].isdigit() or not parts[2].isdigit():
            continue
        rows.append(
            {
                "state": parts[0],
                "recv_q": int(parts[1]),
                "send_q": int(parts[2]),
                "local": parts[3],
                "peer": parts[4],
            }
        )
    return rows


def parse_journal_summary(output: str) -> dict[str, int]:
    summary = _empty_journal_summary()
    for line in output.splitlines():
        if "Started xsignal-vpe-" in line and "stream-daemon.service" in line:
            summary = _empty_journal_summary()
            continue
        if "reconcile_pass" in line and '"status": "clean"' in line:
            summary["reconcile_clean"] += 1
            summary["reconcile_error_since_clean"] = 0
            summary["stream_errors_since_clean"] = 0
            summary["rest_429_since_clean"] = 0
            summary["user_data_stream_errors_since_clean"] = 0
        if "reconcile_pass" in line and '"status": "error"' in line:
            summary["reconcile_error"] += 1
            summary["reconcile_error_since_clean"] += 1
        if '"event": "stream_error"' in line:
            summary["stream_errors"] += 1
            summary["stream_errors_since_clean"] += 1
        if "-1003" in line or " 429 " in line:
            summary["rest_429"] += 1
            summary["rest_429_since_clean"] += 1
        if '"event": "stream_connected"' in line:
            summary["stream_connected"] += 1
        if '"event": "user_data_stream_connected"' in line:
            summary["user_data_stream_connected"] += 1
        if '"event": "user_data_stream_error"' in line:
            summary["user_data_stream_errors"] += 1
            summary["user_data_stream_errors_since_clean"] += 1
        if '"event": "strategy_action"' in line:
            summary["strategy_actions"] += 1
    return summary


def _empty_journal_summary() -> dict[str, int]:
    return {
        "reconcile_clean": 0,
        "reconcile_error": 0,
        "reconcile_error_since_clean": 0,
        "stream_errors": 0,
        "stream_errors_since_clean": 0,
        "user_data_stream_connected": 0,
        "user_data_stream_errors": 0,
        "user_data_stream_errors_since_clean": 0,
        "rest_429": 0,
        "rest_429_since_clean": 0,
        "stream_connected": 0,
        "strategy_actions": 0,
    }


def render_status_text(snapshot: dict[str, object]) -> str:
    lines = [
        f"OVERALL {snapshot['overall']}",
        f"REVISION {snapshot.get('revision') or 'unknown'}",
        (
            "SERVICE "
            f"active={_system_value(snapshot, 'service_active')} "
            f"live_active={_system_value(snapshot, 'live_service_active')} "
            f"live_guard={_system_value(snapshot, 'live_guard_present')}"
        ),
        f"WARNINGS {', '.join(snapshot['warnings']) if snapshot['warnings'] else 'none'}",
        f"SOCKETS {len(snapshot['sockets'])}",
    ]
    for socket in snapshot["sockets"]:
        lines.append(
            "  "
            f"{socket['state']} recv_q={socket['recv_q']} send_q={socket['send_q']} "
            f"peer={socket['peer']}"
        )
    lines.append("MARKET_BARS")
    for interval, item in snapshot["market_bars"].items():
        lines.append(
            f"  {interval} rows={item['rows']} latest_open_time={item['latest_open_time']}"
        )
    lines.append("CURSORS")
    for interval, item in snapshot["cursors"].items():
        lines.append(
            f"  {interval} symbols={item['symbols']} max_lag_seconds={item['max_lag_seconds']}"
        )
    lines.append(f"ACTIVE_POSITIONS {snapshot['positions']['active']}")
    lines.append(f"ERROR_LOCKED_POSITIONS {snapshot['positions']['error_locked']}")
    lines.append(f"UNRESOLVED_ORDER_INTENTS {snapshot['orders']['unresolved']}")
    lines.append(f"EXCHANGE_CONFIRMED_ORDER_INTENTS {snapshot['orders']['exchange_confirmed']}")
    lines.append(f"JOURNAL {json.dumps(snapshot['journal'], sort_keys=True)}")
    return "\n".join(lines)


def _market_bar_summary(conn: sqlite3.Connection) -> dict[str, dict[str, object]]:
    if not _table_exists(conn, "market_bars"):
        return {}
    rows = conn.execute(
        """
        select interval, count(*) as rows, max(open_time) as latest_open_time
        from market_bars
        group by interval
        order by interval
        """
    ).fetchall()
    return {
        row["interval"]: {
            "rows": int(row["rows"]),
            "latest_open_time": row["latest_open_time"],
        }
        for row in rows
    }


def _cursor_summary(conn: sqlite3.Connection, *, now: datetime) -> dict[str, dict[str, object]]:
    if not _table_exists(conn, "market_cursors"):
        return {}
    rows = conn.execute(
        """
        select interval, count(*) as symbols, min(last_open_time) as min_open_time,
               max(last_open_time) as max_open_time
        from market_cursors
        group by interval
        order by interval
        """
    ).fetchall()
    summary = {}
    for row in rows:
        max_open_time = datetime.fromisoformat(row["max_open_time"])
        summary[row["interval"]] = {
            "symbols": int(row["symbols"]),
            "min_open_time": row["min_open_time"],
            "max_open_time": row["max_open_time"],
            "max_lag_seconds": max(int((now - max_open_time).total_seconds()), 0),
        }
    return summary


def _position_summary(conn: sqlite3.Connection) -> dict[str, int]:
    if not _table_exists(conn, "positions"):
        return {"active": 0, "error_locked": 0}
    active = conn.execute(
        """
        select count(*) from positions
        where state in ('OPEN', 'ADD_ARMED', 'ADD_SUBMITTED', 'STOP_REPLACING', 'EXITING')
        """
    ).fetchone()[0]
    error_locked = conn.execute(
        "select count(*) from positions where state = 'ERROR_LOCKED'"
    ).fetchone()[0]
    return {"active": int(active), "error_locked": int(error_locked)}


def _order_summary(conn: sqlite3.Connection) -> dict[str, int]:
    if not _table_exists(conn, "order_intents"):
        return {"unresolved": 0, "exchange_confirmed": 0, "errors": 0}
    unresolved = conn.execute(
        "select count(*) from order_intents where status in ('PENDING_SUBMIT', 'SUBMITTED')"
    ).fetchone()[0]
    exchange_confirmed = conn.execute(
        "select count(*) from order_intents where status = 'EXCHANGE_CONFIRMED'"
    ).fetchone()[0]
    errors = conn.execute("select count(*) from order_intents where status = 'ERROR'").fetchone()[0]
    return {
        "unresolved": int(unresolved),
        "exchange_confirmed": int(exchange_confirmed),
        "errors": int(errors),
    }


def _warnings(
    *,
    system_available: bool,
    service_active: bool,
    live_service_active: bool,
    live_guard_present: bool,
    sockets: list[dict[str, object]],
    journal: dict[str, int],
    positions: dict[str, int],
    orders: dict[str, int],
    cursors: dict[str, dict[str, object]],
) -> list[str]:
    warnings = []
    if system_available:
        if not service_active:
            warnings.append("testnet_service_inactive")
        if live_service_active:
            warnings.append("live_service_active")
        if live_guard_present:
            warnings.append("live_guard_present")
        if any(socket["recv_q"] or socket["send_q"] for socket in sockets):
            warnings.append("socket_queue_nonzero")
        if int(journal.get("rest_429_since_clean", journal.get("rest_429", 0))) > 0:
            warnings.append("recent_rest_429")
        if int(journal.get("stream_errors_since_clean", journal.get("stream_errors", 0))) > 0:
            warnings.append("recent_stream_errors")
        if int(
            journal.get(
                "user_data_stream_errors_since_clean",
                journal.get("user_data_stream_errors", 0),
            )
        ) > 0:
            warnings.append("recent_user_data_stream_errors")
        if int(journal.get("reconcile_error_since_clean", journal.get("reconcile_error", 0))) > 0:
            warnings.append("recent_reconcile_errors")
        if int(journal.get("reconcile_clean", 0)) == 0:
            warnings.append("no_recent_clean_reconcile")
    if positions["error_locked"] > 0:
        warnings.append("error_locked_positions")
    if orders["unresolved"] > 0:
        warnings.append("unresolved_order_intents")
    if cursors.get("1m", {}).get("max_lag_seconds", 0) > 180:
        warnings.append("cursor_lag")
    return warnings


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _systemctl_is_active(service: str, *, runner) -> bool:
    result = runner(["systemctl", "is-active", service], capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() == "active"


def _systemctl_main_pid(service: str, *, runner) -> str:
    return _run_text(["systemctl", "show", "-p", "MainPID", "--value", service], runner=runner).strip()


def _run_text(command: list[str], *, runner) -> str:
    result = runner(command, capture_output=True, text=True)
    return result.stdout or ""


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _system_value(snapshot: dict[str, object], key: str) -> object:
    if not snapshot.get("system_available", True):
        return "unknown"
    return snapshot[key]


def _local_system_unavailable_snapshot(
    *,
    revision_file: Path,
    live_guard_file: Path,
) -> dict[str, object]:
    return {
        "service_active": False,
        "system_available": False,
        "live_service_active": False,
        "live_guard_present": live_guard_file.exists(),
        "revision": _read_text(revision_file),
        "main_pid": None,
        "sockets": [],
        "journal": {},
    }
