"""Platform metadata polling and deadline monitoring for Givemeanode."""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

METADATA_URL = "http://169.254.42.1/v1/events"
METADATA_HEADER = {"Metadata-Flavor": "givemeanode"}


def parse_deadline(val: str | float | int | None) -> float | None:
    """Parse an ISO timestamp, human date string, or epoch seconds into UTC epoch seconds."""
    if not val:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    val = val.strip()
    try:
        return float(val)
    except ValueError:
        pass
    # Support trailing " UTC"
    if val.endswith(" UTC"):
        val = val[:-4].strip() + "+00:00"
    dt = datetime.fromisoformat(val)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class PlatformSentinel:
    """Monitors credit/time deadline and Givemeanode retirement events."""

    def __init__(
        self,
        deadline: float | None = None,
        check_interval: float = 120.0,
        metadata_url: str = METADATA_URL,
    ):
        self.deadline = deadline
        self.check_interval = check_interval
        self.metadata_url = metadata_url
        self._last_check = 0.0

    def check(self, now: float | None = None) -> str | None:
        """Returns a non-empty stop reason string if training should halt, else None."""
        cur = time.time() if now is None else now
        if self.deadline is not None and cur >= self.deadline:
            return f"deadline reached (current {cur:.0f} >= deadline {self.deadline:.0f})"

        if self.check_interval > 0 and (cur - self._last_check) >= self.check_interval:
            self._last_check = cur
            event_warning = self._poll_events()
            if event_warning:
                return f"node event warning: {event_warning}"

        return None

    def _poll_events(self) -> str | None:
        """Query metadata /v1/events for retirement or termination notices."""
        try:
            req = urllib.request.Request(self.metadata_url, headers=METADATA_HEADER)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None  # offline, non-givemeanode host, or endpoint timeout

        # Events can be a dict or list of event objects
        events = data if isinstance(data, list) else data.get("events", [data])
        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_type = str(ev.get("type", "")).lower()
            ev_msg = str(ev.get("message", ev.get("reason", "")))
            if any(k in ev_type for k in ("retire", "stop", "preempt", "terminate")):
                return f"{ev_type}: {ev_msg}" or ev_type
            if ev.get("machine_retires_at") or ev.get("retires_at"):
                retire_time = ev.get("machine_retires_at") or ev.get("retires_at")
                return f"machine_retires_at {retire_time}"

        return None
