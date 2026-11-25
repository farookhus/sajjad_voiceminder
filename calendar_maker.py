"""
calendar_maker.py  —  turns a reminder into something a phone calendar understands.

Two ways to get the reminder into the phone:
  1. make_ics()          -> an .ics file (the universal "calendar invite" format).
                            iPhone Calendar, Google Calendar, Outlook and Samsung
                            Calendar can all open it. It includes an ALARM, so
                            the phone buzzes at the right time.
  2. google_calendar_link() -> a link that opens Google Calendar with the event
                            already filled in. Tap "Save" and it's done.
"""

import datetime as dt
import uuid
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from brain import CATEGORIES

EVENT_LENGTH = dt.timedelta(minutes=30)


def to_utc(local_time, timezone_name):
    """Phones all over the world agree on UTC time, so we convert to it."""
    aware = local_time.replace(tzinfo=ZoneInfo(timezone_name))
    return aware.astimezone(dt.timezone.utc)


def _ics_time(t):
    return t.strftime("%Y%m%dT%H%M%SZ")


def _escape(text):
    # In .ics files, commas, semicolons and backslashes need a "\" in front.
    return (text.replace("\\", "\\\\").replace(";", "\\;")
                .replace(",", "\\,").replace("\n", "\\n"))


def _fold(line):
    """Calendar files want lines of at most 75 bytes; longer ones continue
    on the next line, which starts with a space."""
    out, current = [], ""
    for ch in line:
        if len((current + ch).encode("utf-8")) > 74:
            out.append(current)
            current = " "
        current += ch
    out.append(current)
    return "\r\n".join(out)


def _event_lines(reminder, timezone_name):
    info = CATEGORIES[reminder["category"]]
    start = to_utc(reminder["when"], timezone_name)
    end = start + EVENT_LENGTH
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uuid.uuid4()}@voiceminder",
        f"DTSTAMP:{_ics_time(dt.datetime.now(dt.timezone.utc))}",
        f"DTSTART:{_ics_time(start)}",
        f"DTEND:{_ics_time(end)}",
        f"SUMMARY:{_escape(info['emoji'] + ' ' + reminder['title'])}",
        f"DESCRIPTION:{_escape('Made with VoiceMinder. You said: ' + reminder['heard'])}",
        f"CATEGORIES:{reminder['category']}",
    ]
    if reminder.get("repeat_rule"):
        lines.append(f"RRULE:{reminder['repeat_rule']}")
    # The ALARM: buzz the phone some minutes before (0 = exactly on time)
    lines += [
        "BEGIN:VALARM",
        "ACTION:DISPLAY",
        f"DESCRIPTION:{_escape(reminder['title'])}",
        f"TRIGGER:-PT{info['alert_minutes']}M",
        "END:VALARM",
        "END:VEVENT",
    ]
    return lines


def make_ics(reminders, timezone_name):
    """reminders = a list of reminder dictionaries. Returns the file text."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0",
             "PRODID:-//VoiceMinder//WAICY 2026//EN", "CALSCALE:GREGORIAN",
             "METHOD:PUBLISH"]
    for r in reminders:
        lines += _event_lines(r, timezone_name)
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(l) for l in lines) + "\r\n"   # .ics uses Windows-style line ends


def google_calendar_link(reminder, timezone_name):
    info = CATEGORIES[reminder["category"]]
    start = to_utc(reminder["when"], timezone_name)
    end = start + EVENT_LENGTH
    params = {
        "action": "TEMPLATE",
        "text": f"{info['emoji']} {reminder['title']}",
        "dates": f"{_ics_time(start)}/{_ics_time(end)}",
        "details": f"Made with VoiceMinder. You said: {reminder['heard']}",
    }
    if reminder.get("repeat_rule"):
        params["recur"] = f"RRULE:{reminder['repeat_rule']}"
    return "https://calendar.google.com/calendar/render?" + urlencode(params)
