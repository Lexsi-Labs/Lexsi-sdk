"""Reconnect behaviour for the long-lived event stream (issue #2261).

`poll_events` / `fetch_event_metrics` consume `GET /v2/events/poll` for the
whole lifetime of a job. Proxies, deploys and HTTP/2 GOAWAY frames reset that
connection mid-run (`httpx.ReadError`, Errno 104). The stream must reconnect
with the same event_id and continue without losing progress output.
"""

import httpx
import pytest

from lexsi_sdk.common import utils
from lexsi_sdk.common.utils import fetch_event_metrics, poll_events


class FakeStreamClient:
    """APIClient stand-in whose ``stream()`` replays scripted outcomes.

    Each entry of ``scripts`` is what one ``stream()`` call yields: decoded
    event dicts and/or exceptions to raise mid-iteration. The last entry is
    repeated for any further calls.
    """

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0
        self.uris = []

    def stream(self, uri, method="GET", payload=None):
        self.uris.append(uri)
        script = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1

        def gen():
            for item in script:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return gen()


def running_event(message, progress, logs="", status="running", metrics=None):
    details = {
        "message": message,
        "progress": progress,
        "status": status,
        "event_logs": logs,
    }
    if metrics:
        details.update(metrics)
    return {"success": True, "details": details}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip reconnect backoff so tests run instantly."""
    monkeypatch.setattr(utils.time_module, "sleep", lambda seconds: None)


def test_poll_events_reconnects_after_transport_error(capsys):
    reset = httpx.ReadError("Connection reset by peer")
    client = FakeStreamClient(
        [
            [running_event("starting", 10, logs="line1\n"), reset],
            [running_event("done", 100, logs="line1\nline2\n", status="completed")],
        ]
    )

    poll_events(client, "proj", "evt1", plot=False)

    assert client.calls == 2
    out = capsys.readouterr().out
    assert "starting" in out
    assert "reconnecting" in out
    assert "done" in out
    # log output is cumulative server-side; the client-side cursor must not
    # re-print lines already shown before the drop
    assert out.count("line1") == 1
    assert "line2" in out
    # resumed with the same event_id
    assert all("event_id=evt1" in uri for uri in client.uris)


def test_poll_events_reconnects_when_connection_fails_before_first_event(capsys):
    client = FakeStreamClient(
        [
            [httpx.ConnectError("connection refused")],
            [running_event("done", 100, status="completed")],
        ]
    )

    poll_events(client, "proj", "evt2", plot=False)

    assert client.calls == 2
    assert "reconnecting" in capsys.readouterr().out


def test_poll_events_gives_up_after_max_consecutive_failures():
    client = FakeStreamClient([[httpx.ReadError("Connection reset by peer")]])

    with pytest.raises(httpx.ReadError):
        poll_events(client, "proj", "evt3", plot=False)

    # default cap: 8 consecutive transport failures, no event in between
    assert client.calls == 8


def test_poll_events_does_not_retry_business_failures():
    client = FakeStreamClient([[{"success": False, "details": "No event found"}]])

    with pytest.raises(Exception, match="No event found"):
        poll_events(client, "proj", "evt4", plot=False)

    # a server-reported failure is terminal: exactly one stream attempt
    assert client.calls == 1


def test_poll_events_clean_end_needs_no_reconnect():
    client = FakeStreamClient(
        [[running_event("done", 100, status="completed")]]
    )

    poll_events(client, "proj", "evt5", plot=False)

    assert client.calls == 1


def test_fetch_event_metrics_reconnects_and_returns_series(capsys):
    reset = httpx.ReadError("[Errno 104] Connection reset by peer")
    metrics = {"model_metrics": {"loss": [[1, 0.5]]}}
    client = FakeStreamClient(
        [
            [running_event("training", 50, metrics=metrics), reset],
            [
                running_event(
                    "done",
                    100,
                    status="completed",
                    metrics={"model_metrics": {"loss": [[1, 0.5], [2, 0.3]]}},
                )
            ],
        ]
    )

    result = fetch_event_metrics(client, "proj", "evt6", plot=False)

    # attempt 1: running event then reset, attempt 2: resumed stream
    assert client.calls == 2
    assert result["model_metrics"]["loss"] == [[1, 0.5], [2, 0.3]]


def test_poll_events_failed_status_still_raises_without_retry():
    client = FakeStreamClient(
        [[running_event("boom", 40, status="failed")]]
    )

    with pytest.raises(Exception):
        poll_events(client, "proj", "evt7", plot=False)

    assert client.calls == 1
