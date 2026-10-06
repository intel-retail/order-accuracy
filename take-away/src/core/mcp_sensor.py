"""Local, dependency-free MCP sensor scaffolding (replaces ``mcp_service_sdk``).

Order Accuracy previously built its MCP surface on top of the shared
``mcp-service-sdk`` package (a thin wrapper that bound ``describe``,
``subscribe``, read tools, and gated action tools onto a FastMCP app). That
package pulled in the official ``mcp`` SDK transitively and forced a
``subscribe`` tool onto every service regardless of need (see the removed
``mcp_server.py`` "strip subscribe" workaround this module replaces).

Order Accuracy is a **sensor only** (Issue #102: "A: none (read/detect
only)") — it never needs ``subscribe``/callback or action tools, so rather
than depend on a shared, general-purpose SDK and then undo part of its
surface, this module vendors only the pieces this service actually uses,
built directly on FastMCP:

  - ``EventEnvelope`` / ``new_event``   — the same durable event shape
    (``event_type``, ``service``, ``store_id``, ``payload``, ``ref_id``,
    ``ts_ms``, ``schema_version``) the SDK used, byte-for-byte compatible
    with data already written to an existing event log/volume.
  - ``SQLiteLog`` / ``JSONLFileLog``    — the same durable log backends,
    same on-disk schema, same idempotent-on-``ref_id`` semantics. A
    populated ``order_accuracy_events.db`` from before this migration opens
    and reads back unchanged; nothing here creates or migrates that file's
    schema.
  - ``Delivery`` / ``WebhookSink`` / ``DisabledSink`` — the same optional,
    best-effort webhook fan-out.
  - ``SensorService``                  — a minimal façade (register event
    types, register read tools, emit, describe, build the MCP app) that
    exposes exactly ``describe`` plus the registered read tools over MCP —
    no ``subscribe``, no action tools, by construction rather than by
    post-hoc removal.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Protocol

# ---------------------------------------------------------------------------
# Event envelope (same shape as mcp_service_sdk.envelope.EventEnvelope)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventEnvelope:
    """A single durable, replayable event.

    ``ref_id`` makes replay idempotent: re-emitting the same ``ref_id`` is a
    no-op at the log layer.
    """

    event_type: str
    service: str
    store_id: str
    payload: dict[str, Any]
    ref_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Wall-clock time in epoch milliseconds; ordering is by log sequence, not this.
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    schema_version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "EventEnvelope":
        data = json.loads(raw)
        return cls(**data)


def new_event(
    event_type: str,
    service: str,
    store_id: str,
    payload: dict[str, Any],
    ref_id: str | None = None,
) -> EventEnvelope:
    """Convenience factory so services never build the envelope by hand."""
    kwargs: dict[str, Any] = {
        "event_type": event_type,
        "service": service,
        "store_id": store_id,
        "payload": payload,
    }
    if ref_id is not None:
        kwargs["ref_id"] = ref_id
    return EventEnvelope(**kwargs)


# ---------------------------------------------------------------------------
# Durable log (same contract/schema as mcp_service_sdk.log)
# ---------------------------------------------------------------------------


class DurableLog(Protocol):
    """The storage contract this service depends on. Swappable per deployment."""

    def append(self, event: EventEnvelope) -> int:
        """Persist an event, return its sequence number. Idempotent on ref_id."""
        ...

    def read(
        self,
        event_type: str | None = None,
        since_seq: int = 0,
        limit: int = 1000,
    ) -> list[EventEnvelope]:
        ...

    def replay(self, from_seq: int = 0) -> Iterator[EventEnvelope]:
        """Re-emit events in original order for benchmarking / debugging."""
        ...


class SQLiteLog:
    """Single-box durable log. Thread-safe via a process-local lock.

    Schema is identical to the previous ``mcp_service_sdk.log.SQLiteLog`` so
    an existing ``order_accuracy_events.db`` file continues to open and read
    back exactly as before — this migration never deletes or recreates that
    file/volume.
    """

    def __init__(self, path: str = ":memory:", service: str = "unknown") -> None:
        import sqlite3

        self._service = service
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ref_id       TEXT UNIQUE NOT NULL,
                    event_type   TEXT NOT NULL,
                    ts_ms        INTEGER NOT NULL,
                    envelope     TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)"
            )

    def append(self, event: EventEnvelope) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT seq FROM events WHERE ref_id = ?", (event.ref_id,)
            )
            row = cur.fetchone()
            if row is not None:
                # Idempotent replay: same ref_id already stored.
                return int(row[0])
            cur = self._conn.execute(
                "INSERT INTO events (ref_id, event_type, ts_ms, envelope) "
                "VALUES (?, ?, ?, ?)",
                (event.ref_id, event.event_type, event.ts_ms, event.to_json()),
            )
            return int(cur.lastrowid)

    def read(
        self,
        event_type: str | None = None,
        since_seq: int = 0,
        limit: int = 1000,
    ) -> list[EventEnvelope]:
        query = "SELECT envelope FROM events WHERE seq > ?"
        params: list[object] = [since_seq]
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY seq ASC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [EventEnvelope.from_json(r[0]) for r in rows]

    def replay(self, from_seq: int = 0) -> Iterator[EventEnvelope]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT envelope FROM events WHERE seq > ? ORDER BY seq ASC",
                (from_seq,),
            ).fetchall()
        for r in rows:
            yield EventEnvelope.from_json(r[0])

    def close(self) -> None:
        self._conn.close()


class JSONLFileLog:
    """File-backed durable log (one JSON object per line).

    Same ``DurableLog`` contract/on-disk record shape as the previous
    ``mcp_service_sdk.log.JSONLFileLog``: ordered by ``seq``, idempotent on
    ``ref_id``, restart-safe (state is rebuilt from the file on open),
    replayable.
    """

    def __init__(self, path: str, service: str = "unknown") -> None:
        self._service = service
        self._path = path
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._seq = 0
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._rebuild_state()

    def _rebuild_state(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._seq = max(self._seq, int(rec["seq"]))
                self._seen.add(rec["event"]["ref_id"])

    def append(self, event: EventEnvelope) -> int:
        with self._lock:
            if event.ref_id in self._seen:
                return self._seq_of(event.ref_id)
            self._seq += 1
            rec = {"seq": self._seq, "event": json.loads(event.to_json())}
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._seen.add(event.ref_id)
            return self._seq

    def _seq_of(self, ref_id: str) -> int:
        for rec in self._iter_records():
            if rec["event"]["ref_id"] == ref_id:
                return int(rec["seq"])
        return 0

    def _iter_records(self) -> Iterator[dict]:
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def read(
        self,
        event_type: str | None = None,
        since_seq: int = 0,
        limit: int = 1000,
    ) -> list[EventEnvelope]:
        out: list[EventEnvelope] = []
        with self._lock:
            for rec in self._iter_records():
                if int(rec["seq"]) <= since_seq:
                    continue
                ev = rec["event"]
                if event_type is not None and ev["event_type"] != event_type:
                    continue
                out.append(EventEnvelope(**ev))
                if len(out) >= limit:
                    break
        return out

    def replay(self, from_seq: int = 0) -> Iterator[EventEnvelope]:
        with self._lock:
            records = [r for r in self._iter_records() if int(r["seq"]) > from_seq]
        for rec in records:
            yield EventEnvelope(**rec["event"])


# ---------------------------------------------------------------------------
# Delivery fan-out (same contract as mcp_service_sdk.delivery)
# ---------------------------------------------------------------------------


class Sink(Protocol):
    name: str

    def push(self, event: EventEnvelope) -> bool:
        """Deliver one event. Return True on success, False to trigger retry."""
        ...


class DisabledSink:
    """Drops delivery on purpose — used for clean benchmark runs."""

    name = "disabled"

    def push(self, event: EventEnvelope) -> bool:  # noqa: ARG002 - intentional no-op
        return True


class WebhookSink:
    """POSTs the event envelope to a callback URL (stdlib, no extra deps)."""

    name = "webhook"

    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self._url = url
        self._timeout_s = timeout_s

    def push(self, event: EventEnvelope) -> bool:
        import urllib.error
        import urllib.request

        data = event.to_json().encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError):
            return False


class Delivery:
    """Fans an event out to all enabled sinks with bounded retries."""

    def __init__(
        self,
        sinks: list[Sink] | None = None,
        max_retries: int = 3,
        backoff_s: float = 0.5,
    ) -> None:
        self._sinks: list[Sink] = sinks or [DisabledSink()]
        self._max_retries = max_retries
        self._backoff_s = backoff_s

    def dispatch(self, event: EventEnvelope) -> dict[str, bool]:
        """Deliver to every sink; returns per-sink final success flag."""
        results: dict[str, bool] = {}
        for sink in self._sinks:
            results[sink.name] = self._push_with_retry(sink, event)
        return results

    def _push_with_retry(self, sink: Sink, event: EventEnvelope) -> bool:
        for attempt in range(self._max_retries):
            if sink.push(event):
                return True
            if attempt < self._max_retries - 1:
                time.sleep(self._backoff_s * (2**attempt))
        return False


# ---------------------------------------------------------------------------
# SensorService — the minimal MCP scaffolding a read/detect-only service needs
# ---------------------------------------------------------------------------


@dataclass
class _ReadTool:
    name: str
    fn: Callable[..., Any]
    description: str
    schema: dict[str, Any]


@dataclass
class SensorService:
    """One per service. Domain code registers event types + read tools; this
    builds ``describe`` and the FastMCP app.

    Deliberately has no ``subscribe``, no act tools, and no policy gate:
    Order Accuracy is a sensor (Issue #102 — "A: none (read/detect only)"),
    so the MCP surface built by :meth:`to_mcp` is exactly ``describe`` plus
    the registered read tools, with nothing to strip after the fact.
    """

    service: str
    store_id: str
    log: DurableLog

    _event_types: dict[str, dict] = field(default_factory=dict)
    _read_tools: dict[str, _ReadTool] = field(default_factory=dict)

    def register_event_type(self, name: str, schema: dict[str, Any]) -> None:
        self._event_types[name] = schema

    def read_tool(
        self, name: str, description: str | None = None, schema: dict | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a read tool exposed over MCP.

        If ``description`` is omitted, the function's docstring is used (the
        standard MCP convention).
        """

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            import inspect

            desc = description if description is not None else (inspect.getdoc(fn) or "")
            self._read_tools[name] = _ReadTool(name, fn, desc, schema or {})
            return fn

        return deco

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        ref_id: str | None = None,
    ) -> EventEnvelope:
        """Emit-to-log-first, then fan out to enabled sinks."""
        event = new_event(event_type, self.service, self.store_id, payload, ref_id)
        self.log.append(event)
        return event

    def describe(self) -> dict[str, Any]:
        """Self-description rich enough for a coding agent to use unassisted.

        ``act_tools`` is always empty: this service never registers action
        tools, by construction (Order Accuracy is read/detect only).
        """
        return {
            "service": self.service,
            "store_id": self.store_id,
            "event_types": self._event_types,
            "read_tools": {
                t.name: {"description": t.description, "schema": t.schema}
                for t in self._read_tools.values()
            },
            "act_tools": {},
        }

    def to_mcp(self):
        """Build a FastMCP app exposing ``describe`` plus the registered read
        tools — nothing else. No ``subscribe``/callback tool is ever bound,
        so there is nothing to remove post-construction.
        """
        from fastmcp import FastMCP

        app = FastMCP(self.service)

        @app.tool(name="describe", description="Describe this service's contract.")
        def _describe() -> dict:
            return self.describe()

        for tool in self._read_tools.values():
            app.tool(name=tool.name, description=tool.description)(
                self._wrap_read(tool)
            )

        return app

    def _wrap_read(self, tool: _ReadTool) -> Callable[..., Any]:
        """Wrap a read tool, preserving its signature so FastMCP still
        derives the input schema from the original function's typed
        parameters (``functools.wraps`` sets ``__wrapped__``).
        """

        @functools.wraps(tool.fn)
        def wrapper(**args: Any) -> Any:
            return tool.fn(**args)

        return wrapper
