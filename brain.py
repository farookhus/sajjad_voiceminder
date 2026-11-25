"""
brain.py  —  VoiceMinder's "brain"

Turns one spoken sentence like
    "remind me to take my medicine every day at 8 pm"
into a structured reminder:
    title="Take my medicine", when=today 8:00 PM, repeat="Every day",
    category="Medicine" (93% sure)

Three AI / smart pieces live here:
  1. understand_time()  -> finds the date and time inside the sentence
  2. find_repeat()      -> spots "every day", "every Monday", "every year" ...
  3. train_classifier() -> a machine-learning model WE TRAIN that guesses
                           the category (Medicine, Study, Birthday ...)
"""

import re
import datetime as dt

import pandas as pd
import parsedatetime as pdt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------------------------
# Category settings: emoji, colour, and how early the phone should alert you
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Medicine":    {"emoji": "💊", "color": "#E53935", "alert_minutes": 0},
    "Study":       {"emoji": "📚", "color": "#1E88E5", "alert_minutes": 30},
    "Appointment": {"emoji": "🩺", "color": "#8E24AA", "alert_minutes": 60},
    "Birthday":    {"emoji": "🎂", "color": "#F4511E", "alert_minutes": 1440},  # 1 day early
    "Chores":      {"emoji": "🧹", "color": "#43A047", "alert_minutes": 10},
    "Shopping":    {"emoji": "🛒", "color": "#FB8C00", "alert_minutes": 15},
    "Activity":    {"emoji": "⚽", "color": "#00897B", "alert_minutes": 30},
    "Other":       {"emoji": "📌", "color": "#546E7A", "alert_minutes": 10},
}

# If the model is less sure than this, we say "Other" instead of guessing.
CONFIDENCE_THRESHOLD = 0.30

DEFAULT_HOUR = 9  # used when the user says a day but no time ("on Friday")


# ---------------------------------------------------------------------------
# 1. The machine-learning model (category guesser)
# ---------------------------------------------------------------------------
def train_classifier(csv_path="data/training_sentences.csv"):
    """Reads example sentences and teaches a model which words go with
    which category. Returns the trained model."""
    data = pd.read_csv(csv_path)
    model = make_pipeline(
        # Turn each sentence into numbers: which words appear,
        # and how important each word is (rare, special words count more).
        TfidfVectorizer(lowercase=True),
        # Learn a "weight" for every word for every category.
        LogisticRegression(max_iter=1000, C=10),
    )
    model.fit(data["sentence"], data["category"])
    return model


def guess_category(model, sentence):
    """Returns (category, confidence between 0 and 1)."""
    probabilities = model.predict_proba([sentence])[0]
    best = probabilities.argmax()
    category = model.classes_[best]
    confidence = float(probabilities[best])
    if confidence < CONFIDENCE_THRESHOLD:
        return "Other", confidence
    return category, confidence


# ---------------------------------------------------------------------------
# 2. Repeats: "every day", "every Monday", "daily", "every year" ...
# ---------------------------------------------------------------------------
def find_repeat(text):
    """Returns (rule, label, text_without_repeat_words).
    rule is in calendar language (RRULE), e.g. 'FREQ=DAILY'."""
    t = text
    # every Monday / every Saturday ...
    m = re.search(r"\b(?:every|each)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", t, re.I)
    if m:
        day = m.group(1).lower()
        # keep the weekday so the time-finder knows which day to start on
        t = t[:m.start()] + day + t[m.end():]
        # The first reminder lands on that weekday, so "every week" from
        # there repeats on the same weekday.
        return "FREQ=WEEKLY", f"Every {day.title()}", t
    patterns = [
        (r"\b(?:every\s*day|everyday|daily|each day)\b", "FREQ=DAILY", "Every day"),
        (r"\b(?:every|each)\s+(?:night|morning|evening)\b", "FREQ=DAILY", "Every day"),
        (r"\b(?:every|each)\s+week(?:end)?\b|\bweekly\b", "FREQ=WEEKLY", "Every week"),
        (r"\b(?:every|each)\s+month\b|\bmonthly\b", "FREQ=MONTHLY", "Every month"),
        (r"\b(?:every|each)\s+year\b|\byearly\b|\bannually\b", "FREQ=YEARLY", "Every year"),
    ]
    for pattern, rule, label in patterns:
        m = re.search(pattern, t, re.I)
        if m:
            # "every morning" still tells us the time of day, so keep "morning"
            kept = ""
            word = m.group(0).lower()
            for part in ("morning", "evening", "night"):
                if part in word:
                    kept = "tonight" if part == "night" else f"in the {part}"
            t = t[:m.start()] + kept + t[m.end():]
            return rule, label, t
    return None, "Does not repeat", t


# ---------------------------------------------------------------------------
# 3. Date & time understanding
# ---------------------------------------------------------------------------
_calendar = pdt.Calendar(version=pdt.VERSION_CONTEXT_STYLE)


MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def _tidy_spoken_time(text, now):
    """Speech-to-text writes times in many ways. Make them easy to read."""
    t = " " + text.strip() + " "
    t = re.sub(r"\b(?:the )?day after tomorrow\b", "in 2 days", t, flags=re.I)

    # "on the 5th" (no month said) -> "on 5 December" (this month, or next month)
    def add_month(m):
        day = int(m.group(1))
        month, year = now.month, now.year
        if day < now.day:
            month, year = (1, year + 1) if month == 12 else (month + 1, year)
        return f"{m.group(0).split('the')[0]}{day} {MONTHS[month - 1]} {year}"
    month_names = "|".join(MONTHS) + "|" + "|".join(x[:3] for x in MONTHS)
    t = re.sub(r"\b(?:on |by )?the (\d{1,2})(?:st|nd|rd|th)\b(?!\s*(?:of\s+)?(?:" + month_names + r"))",
               add_month, t, flags=re.I)
    t = re.sub(r"\b([ap])\.\s?m\.?", r"\1m", t, flags=re.I)          # p.m. -> pm
    t = re.sub(r"\bnoon\b", "12 pm", t, flags=re.I)
    t = re.sub(r"\bmidnight\b", "11:59 pm", t, flags=re.I)
    # "9 in the morning" -> "9 am",  "6 in the evening" -> "6 pm"
    t = re.sub(r"\b(\d{1,2}(?::\d{2})?)\s+in the morning\b", r"\1 am", t, flags=re.I)
    t = re.sub(r"\b(\d{1,2}(?::\d{2})?)\s+(?:in the (?:afternoon|evening)|at night|tonight)\b",
               r"\1 pm", t, flags=re.I)
    # time-of-day words on their own -> a sensible clock time
    t = re.sub(r"\bin the morning\b|\bmorning\b", "at 8 am", t, flags=re.I)
    t = re.sub(r"\bin the afternoon\b|\bafternoon\b", "at 3 pm", t, flags=re.I)
    t = re.sub(r"\bin the evening\b|\bevening\b", "at 6 pm", t, flags=re.I)

    # "at 6" with no am/pm: our rule -> 1 to 7 means PM, 8 to 11 means AM
    def fix_bare_hour(m):
        hour = int(m.group(1))
        rest = m.group(2) or ""
        suffix = "pm" if 1 <= hour <= 7 else "am"
        return f"at {hour}{rest} {suffix}"
    t = re.sub(r"\bat (\d{1,2})(:\d{2})?\b(?!\s*(?:am|pm|:))", fix_bare_hour, t, flags=re.I)
    return t.strip()


def understand_time(text, now):
    """Finds the date/time in the sentence.
    now = the current local time (a datetime without timezone).
    Returns (when, time_was_heard, matched_phrases, tidy_text)."""
    tidy = _tidy_spoken_time(text, now)
    matches = _calendar.nlp(tidy, sourceTime=now) or ()

    date_part, time_part, relative = None, None, None
    phrases = []
    for when, ctx, _start, _end, phrase in matches:
        phrases.append(phrase)
        if re.search(r"\bin \d+ (?:minute|min|hour|hr)", phrase, re.I):
            relative = when               # "in 2 hours" -> exact moment
        if ctx.hasDate and date_part is None:
            date_part = when.date()
        if ctx.hasTime and time_part is None:
            time_part = when.time()

    if relative is not None:
        return relative.replace(second=0, microsecond=0), True, phrases, tidy

    time_heard = time_part is not None
    if time_part is None:
        time_part = dt.time(DEFAULT_HOUR, 0)

    if date_part is None:
        date_part = now.date()
        when = dt.datetime.combine(date_part, time_part)
        if when <= now:                   # "at 7 am" but it's already 10 am
            when += dt.timedelta(days=1)
    else:
        when = dt.datetime.combine(date_part, time_part)
        if when < now and date_part == now.date():
            when += dt.timedelta(days=1)
    return when.replace(second=0, microsecond=0), time_heard, phrases, tidy


# ---------------------------------------------------------------------------
# 4. A clean title: remove "remind me to", the time words, etc.
# ---------------------------------------------------------------------------
FILLER = [
    r"\bhey\b", r"\bplease\b", r"\bcan you\b", r"\bcould you\b",
    r"\bset (?:a |an )?(?:reminder|alarm) (?:to|for|that)\b", r"\bset (?:a |an )?reminder\b",
    r"\bremind me (?:to|about|that)\b", r"\bremind me\b",
    r"\bdon'?t (?:let me )?forget (?:to|about)\b", r"\bi (?:need|have|want) to\b",
    r"\bi should\b",
]
TIME_WORDS = (r"\b(?:today|tonight|tomorrow|day after tomorrow|this|next|on|at|by|in|"
              r"the|morning|afternoon|evening|night|am|pm)\b")


def make_title(original, phrases):
    t = original
    for phrase in phrases:
        t = re.sub(re.escape(phrase), " ", t, flags=re.I)
    for pattern in FILLER:
        t = re.sub(pattern, " ", t, flags=re.I)
    t = re.sub(r"\b(?:every|each)\s+\w+\b|\bdaily\b|\bweekly\b|\bmonthly\b|\byearly\b", " ", t, flags=re.I)
    t = re.sub(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.)", " ", t, flags=re.I)
    t = re.sub(r"\b(?:in the morning|in the evening|in the afternoon|at night|tonight|tomorrow|today)\b",
               " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+,", ",", t).strip(" ,.!?")
    # remove time words left dangling at the end: "call grandma at" -> "call grandma"
    while True:
        new = re.sub(TIME_WORDS + r"$", "", t, flags=re.I).strip(" ,.")
        if new == t:
            break
        t = new
    if not t:
        t = "Reminder"
    return t[0].upper() + t[1:]


# ---------------------------------------------------------------------------
# 5. Put it all together
# ---------------------------------------------------------------------------
def understand(sentence, model, now):
    """The main function: sentence in -> reminder dictionary out."""
    rule, repeat_label, without_repeat = find_repeat(sentence)
    when, time_heard, phrases, tidy = understand_time(without_repeat, now)
    title = make_title(tidy, phrases)
    # We give the model the clean title, so time words like "on" and "at"
    # don't confuse it.
    category, confidence = guess_category(model, title)
    if rule is None and re.search(r"\b(?:birthday|anniversary)\b", sentence, re.I):
        rule, repeat_label = "FREQ=YEARLY", "Every year"   # birthdays come back!
    return {
        "heard": sentence,
        "title": title,
        "when": when,
        "time_heard": time_heard,
        "repeat_rule": rule,
        "repeat_label": repeat_label,
        "category": category,
        "confidence": confidence,
    }


if __name__ == "__main__":
    # Quick test:  python brain.py
    m = train_classifier()
    now = dt.datetime.now().replace(second=0, microsecond=0)
    for s in ["remind me to take my medicine every day at 8 pm",
              "call grandma tomorrow at 5:30 pm",
              "Mom's birthday on December 12"]:
        r = understand(s, m, now)
        print(f"{s!r}\n   -> {r['title']} | {r['when']:%a %d %b %I:%M %p} | "
              f"{r['repeat_label']} | {r['category']} ({r['confidence']:.0%})")
