# 🎙️ VoiceMinder

**Say it once. It lands in your phone's calendar — with an alarm.**

VoiceMinder is a phone web-app that turns a spoken sentence like
*"Remind me to take my medicine every day at 8 pm"* into a real calendar
event with an alarm, in Google Calendar, iPhone Calendar, Samsung Calendar or Outlook.

Built by Sajjad for WAICY 2026 (AI Showcase Track, Middle School) ·
Shree Radhe Coding Sessions with Simran.

## How it works

```
 🎤 voice  ──►  Speech-to-Text (Google)  ──►  "take my medicine every day at 8 pm"
                                                   │
                         ┌─────────────────────────┼─────────────────────────┐
                         ▼                         ▼                         ▼
               Time finder (parsedatetime)  Repeat finder (rules)   Category AI (our own
               "8 pm today"                 "every day"             scikit-learn model)
                                                                    "Medicine 💊, 95% sure"
                         └─────────────────────────┼─────────────────────────┘
                                                   ▼
                              ✏️ The user checks and fixes it  (a human stays in charge)
                                                   ▼
                     📲 .ics calendar file with alarm   +   📅 Google Calendar link
```

## Files

| File | What it does |
|---|---|
| `app.py` | The app screen (Streamlit + our own HTML/CSS card design) |
| `brain.py` | Understands the sentence: time, repeat, title, category |
| `calendar_maker.py` | Makes the `.ics` calendar file and the Google Calendar link |
| `train_model.py` | Tests our category AI and prints its accuracy + confusion matrix |
| `data/training_sentences.csv` | Example sentences the AI learns from (add more!) |
| `data/test_sentences.csv` | Sentences the AI has never seen — used only for testing |

## Run it on a laptop

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
python train_model.py         # how accurate is the AI?
python brain.py               # quick test of the brain
```

## Put it on a phone

1. Push this folder to a GitHub repository.
2. Go to share.streamlit.io → **Create app** → pick the repo → main file `app.py` → **Deploy**.
3. Open the app link on the phone → Chrome menu **Add to Home screen** (Android) or
   Safari **Share → Add to Home Screen** (iPhone). It now opens like an app.

## Privacy & honesty

- The voice clip goes to Google's speech service to become text. VoiceMinder does not store recordings.
- The AI can be wrong, so every reminder is shown for checking before it is added.
- When the AI is less than 30% sure of the category, it says "📌 Other" instead of guessing.
