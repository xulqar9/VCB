"""The user's clock: current date/time in their time zone, and human phrasing
of times ("3 days ago", "late at night").

Time zone: an explicit IANA name (VCB_TIMEZONE, e.g. "Europe/Berlin") or,
by default, whatever this PC uses -- including its daylight-saving rules, since
Python asks the OS for the offset at each moment.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones


def list_timezones() -> list[str]:
    """IANA zone names for the settings dropdown ("Europe/Berlin", ...). Empty
    if the tz database isn't available (Windows without the tzdata package)."""
    try:
        zones = available_timezones()
    except Exception:
        return []
    return sorted(z for z in zones if "/" in z and not z.startswith(("Etc/", "SystemV/")))


def valid_timezone(name: str) -> bool:
    try:
        ZoneInfo(name)
        return True
    except (ZoneInfoNotFoundError, ValueError):
        return False


class Clock:
    def __init__(self, timezone: str | None = None) -> None:
        name = (timezone or "").strip()
        self._zone: tzinfo | None = None
        self.zone_name = ""
        if name:
            try:
                self._zone = ZoneInfo(name)
                self.zone_name = name
            except (ZoneInfoNotFoundError, ValueError):
                pass            # unknown name: fall back to the PC's zone

    @property
    def explicit(self) -> bool:
        return self._zone is not None

    def now(self) -> datetime:
        """Timezone-aware 'now' for the user."""
        if self._zone is not None:
            return datetime.now(self._zone)
        return datetime.now().astimezone()

    def from_timestamp(self, ts: float) -> datetime:
        if self._zone is not None:
            return datetime.fromtimestamp(ts, self._zone)
        return datetime.fromtimestamp(ts).astimezone()

    def localize(self, naive: datetime) -> datetime:
        """A wall-clock time in the user's zone (e.g. "Monday 08:00") -> aware."""
        if self._zone is not None:
            return naive.replace(tzinfo=self._zone)
        return naive.astimezone()       # OS rules for that date (DST-correct)

    def offset_label(self, dt: datetime | None = None) -> str:
        off = (dt or self.now()).utcoffset() or timedelta(0)
        minutes = int(off.total_seconds() // 60)
        sign = "+" if minutes >= 0 else "-"
        hh, mm = divmod(abs(minutes), 60)
        return f"UTC{sign}{hh:02d}:{mm:02d}"

    def zone_label(self, dt: datetime | None = None) -> str:
        """"Europe/Berlin, UTC+02:00" or just "UTC+02:00" in automatic mode."""
        off = self.offset_label(dt)
        return f"{self.zone_name}, {off}" if self.zone_name else off

    # -- phrasing ------------------------------------------------------------

    def now_line(self) -> str:
        """The line given to the model every turn so it never guesses the date.
        Framed as background: models otherwise bring the time up in every reply
        ("at this hour...")."""
        now = self.now()
        return (f"Background, for your own reference only: it is "
                f"{spoken_datetime(now)} ({now:%Y-%m-%d}, {self.zone_label(now)}) "
                "for the user. Don't mention the date or time unless they ask or it "
                "actually matters (a deadline, a plan, a reminder).")

    def ago(self, ts: float, now: datetime | None = None) -> str:
        """How long ago a moment was, the way a person would say it."""
        now = now or self.now()
        then = self.from_timestamp(ts)
        secs = (now - then).total_seconds()
        if secs < 90:
            return "just now"
        if secs < 3600:
            return f"{round(secs / 60)} minutes ago"
        days = (now.date() - then.date()).days
        if days == 0:
            hours = round(secs / 3600)
            return "an hour ago" if hours <= 1 else f"{hours} hours ago"
        if days == 1:
            part = part_of_day(then)
            return "last night" if part in ("night", "late at night") else f"yesterday {part}"
        if days < 7:
            return f"{days} days ago ({then:%A})"
        if days < 60:
            weeks = round(days / 7)
            return "a week ago" if weeks <= 1 else f"{weeks} weeks ago"
        return f"on {then.day} {then:%B %Y}"


def spoken_datetime(dt: datetime) -> str:
    """'Saturday, 24 October 2026, 1:53 AM' -- weekday and month name, so the
    model can't confuse day/month order between countries."""
    clock = dt.strftime("%I:%M %p").lstrip("0")
    return f"{dt:%A}, {dt.day} {dt:%B %Y}, {clock}"


def part_of_day(dt: datetime) -> str:
    h = dt.hour
    if 5 <= h < 8:
        return "early morning"
    if 8 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    if 22 <= h or h < 2:
        return "night"
    return "late at night"


def timestamp() -> float:
    return time.time()
