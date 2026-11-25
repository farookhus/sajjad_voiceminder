"""
app.py  —  VoiceMinder: say it, and it lands in your phone's calendar.

Run on your laptop:     streamlit run app.py
Run on your phone:      deploy free on Streamlit Community Cloud (see the guide),
                        open the link in Chrome/Safari -> "Add to Home Screen".
"""

import datetime as dt
import hashlib
import io
from zoneinfo import ZoneInfo

import speech_recognition as sr
import streamlit as st

from brain import CATEGORIES, train_classifier, understand
from calendar_maker import google_calendar_link, make_ics

# ---------------------------------------------------------------------------
# Settings you can change
# ---------------------------------------------------------------------------
TIMEZONES = ["Asia/Kolkata", "America/New_York", "America/Chicago",
             "America/Denver", "America/Los_Angeles", "Europe/London",
             "Asia/Dubai", "Asia/Singapore", "Australia/Sydney"]
LANGUAGES = {"English (India)": "en-IN", "English (US)": "en-US",
             "English (UK)": "en-GB", "Hindi": "hi-IN"}

st.set_page_config(page_title="VoiceMinder", page_icon="🎙️", layout="centered")

# ---------------------------------------------------------------------------
# Our own HTML/CSS design for the reminder card
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  .vm-title   { font-size: 2.1rem; font-weight: 800; margin-bottom: 0; }
  .vm-sub     { color: #666; margin-top: 0; }
  .vm-card    { border-radius: 18px; padding: 18px 20px; margin: 12px 0;
                color: white; box-shadow: 0 4px 14px rgba(0,0,0,.15); }
  .vm-emoji   { font-size: 2.4rem; float: right; }
  .vm-what    { font-size: 1.35rem; font-weight: 700; }
  .vm-when    { font-size: 1.05rem; margin-top: 4px; }
  .vm-tag     { display: inline-block; background: rgba(255,255,255,.25);
                border-radius: 999px; padding: 2px 10px; margin: 8px 6px 0 0;
                font-size: .85rem; }
  .vm-heard   { font-style: italic; color: #555; }
</style>
""", unsafe_allow_html=True)


def reminder_card(r):
    info = CATEGORIES[r["category"]]
    when = r["when"].strftime("%A, %d %B · %I:%M %p")
    return f"""
    <div class="vm-card" style="background:{info['color']}">
      <span class="vm-emoji">{info['emoji']}</span>
      <div class="vm-what">{r['title']}</div>
      <div class="vm-when">⏰ {when}</div>
      <span class="vm-tag">🔁 {r['repeat_label']}</span>
      <span class="vm-tag">🏷️ {r['category']} · AI is {r['confidence']:.0%} sure</span>
    </div>"""


# ---------------------------------------------------------------------------
# The AI parts
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    # Trains in under a second; re-trains automatically when the app restarts,
    # so new sentences added to the CSV are learned.
    return train_classifier("data/training_sentences.csv")


def speech_to_text(wav_bytes, language):
    """Sends the recording to Google's free speech recognizer."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio, language=language)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown('<div class="vm-title">🎙️ VoiceMinder</div>', unsafe_allow_html=True)
st.markdown('<p class="vm-sub">Say it once. It lands in your phone’s calendar — with an alarm.</p>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    timezone_name = st.selectbox("My time zone", TIMEZONES)
    language = LANGUAGES[st.selectbox("I speak", list(LANGUAGES))]
    st.caption("🔒 Privacy: your voice clip is sent to Google's speech service to turn it "
               "into text. VoiceMinder does not save your recordings.")

if "saved" not in st.session_state:
    st.session_state.saved = []          # the user's reminder list
if "transcripts" not in st.session_state:
    st.session_state.transcripts = {}    # so we don't transcribe the same clip twice

model = get_model()
now = dt.datetime.now(ZoneInfo(timezone_name)).replace(tzinfo=None, second=0, microsecond=0)

# 1) Listen ------------------------------------------------------------------
st.subheader("1 · Say your reminder")
st.caption('Try: "Remind me to take my medicine every day at 8 pm"')
clip = st.audio_input("Tap the mic, speak, then tap stop")
typed = st.text_input("…or type it instead")

sentence = ""
if clip is not None:
    wav = clip.getvalue()
    key = hashlib.md5(wav).hexdigest()
    if key not in st.session_state.transcripts:
        with st.spinner("Listening…"):
            try:
                st.session_state.transcripts[key] = speech_to_text(wav, language)
            except sr.UnknownValueError:
                st.session_state.transcripts[key] = ""
                st.warning("I couldn't make out any words. Try again a little closer to the mic.")
            except sr.RequestError:
                st.session_state.transcripts[key] = ""
                st.error("The speech service is not reachable right now. You can type instead.")
    sentence = st.session_state.transcripts[key]
if typed.strip():
    sentence = typed.strip()

if not sentence:
    st.info("Your reminder will appear here.")
else:
    st.markdown(f'<p class="vm-heard">I heard: “{sentence}”</p>', unsafe_allow_html=True)
    r = understand(sentence, model, now)

    # 2) Check ---------------------------------------------------------------
    st.subheader("2 · Check it")
    if not r["time_heard"]:
        st.warning(f"I didn't hear a time, so I picked {r['when']:%I:%M %p}. Change it below if needed.")
    if r["category"] == "Other":
        st.info("I'm not sure which category this is, so I used 📌 Other.")

    card_slot = st.empty()        # the card goes here, above the "fix it" box
    with st.expander("✏️ Fix anything the AI got wrong"):
        r["title"] = st.text_input("What", r["title"])
        c1, c2 = st.columns(2)
        day = c1.date_input("Date", r["when"].date())
        clock = c2.time_input("Time", r["when"].time(), step=300)
        r["when"] = dt.datetime.combine(day, clock)
        names = list(CATEGORIES)
        r["category"] = st.selectbox("Category", names, index=names.index(r["category"]))
        repeats = {"Does not repeat": None, "Every day": "FREQ=DAILY",
                   "Every week": "FREQ=WEEKLY", "Every month": "FREQ=MONTHLY",
                   "Every year": "FREQ=YEARLY"}
        if r["repeat_label"] not in repeats:          # e.g. "Every Saturday"
            repeats[r["repeat_label"]] = r["repeat_rule"]
        choice = st.selectbox("Repeat", list(repeats), index=list(repeats).index(r["repeat_label"]))
        r["repeat_label"], r["repeat_rule"] = choice, repeats[choice]

    card_slot.markdown(reminder_card(r), unsafe_allow_html=True)

    # 3) Add to phone ---------------------------------------------------------
    st.subheader("3 · Add it to my phone")
    b1, b2 = st.columns(2)
    b1.download_button("📲 Add to phone calendar (.ics)",
                       data=make_ics([r], timezone_name),
                       file_name="reminder.ics", mime="text/calendar",
                       use_container_width=True)
    b2.link_button("📅 Open in Google Calendar",
                   google_calendar_link(r, timezone_name), use_container_width=True)
    if st.button("➕ Save to my list", use_container_width=True):
        st.session_state.saved.append(dict(r))
        st.success("Saved!")

# My list ------------------------------------------------------------------------
if st.session_state.saved:
    st.divider()
    st.subheader(f"🗂️ My reminders ({len(st.session_state.saved)})")
    for item in sorted(st.session_state.saved, key=lambda x: x["when"]):
        info = CATEGORIES[item["category"]]
        st.write(f"{info['emoji']} **{item['title']}** — {item['when']:%a %d %b, %I:%M %p}"
                 f" · {item['repeat_label']}")
    st.download_button("📲 Add ALL to phone calendar",
                       data=make_ics(st.session_state.saved, timezone_name),
                       file_name="my_reminders.ics", mime="text/calendar")
