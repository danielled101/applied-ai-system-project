# 🎵 Music Recommender Simulation

## Project Summary

A small music recommender system that scores a song catalog against a user's
taste profile and explains each recommendation. Explanations are generated
with **Retrieval-Augmented Generation (RAG)**: before calling Gemini, the
system retrieves (1) the specific preference-matching facts that drove a
song's score and (2) the catalog's nearest neighbors to that song by audio
attributes. Gemini is then instructed to write the explanation using *only*
that retrieved context, so the response is grounded rather than invented. If
no API key is configured, the app falls back to a deterministic template
built from the same retrieved facts, so it always runs end-to-end.

---

## How The System Works

- **`Song`** attributes: genre, mood, energy, tempo_bpm, valence,
  danceability, acousticness.
- **`UserProfile`** stores: favorite_genre, favorite_mood, target_energy,
  likes_acoustic.
- **Scoring** (`score_song`): rewards genre match, mood match, closeness to
  target energy, and acousticness alignment, returning both a numeric score
  and the human-readable reasons behind it.
- **Retrieval** (`retrieve_similar_songs`): finds each recommended song's
  nearest catalog neighbors by attribute distance (energy, valence,
  danceability, acousticness, tempo).
- **Generation** (`generate_explanation`): sends the score reasons + the
  retrieved neighbors to Gemini, which writes a short, grounded explanation.
  Falls back to a template built from the same facts if no API key is set
  or the call fails.
- See [`diagrams/architecture.mmd`](diagrams/architecture.mmd) for the full
  data flow diagram.

---

## AI Feature: RAG-Grounded Explanations

This project's advanced AI feature is **Retrieval-Augmented Generation**.
`explain_recommendation()` / `recommend_songs()` never let the model
free-write — they always retrieve grounding facts first (matched
preferences + similar-song neighbors) and pass those into the prompt, with
an explicit instruction not to invent attributes. This is fully integrated
into the main recommendation path, not a standalone script: every call to
`recommend_songs` produces its explanation this way.

**Guardrails:** missing API key, missing `google-genai` package, API errors,
timeouts, and empty responses are all caught and logged, with an automatic
fallback to a deterministic template — the app never crashes because of the
LLM call.

**Logging:** all runs log to both the console and `logs/app.log` (catalog
loads, retrieval hits, API failures/fallbacks).

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. (Optional, for AI-generated explanations) Create a `.env` file in the
   project root with your Gemini API key (get one free at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey)):

```
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.0-flash
```

Without a key, the app still runs — explanations fall back to a
deterministic template built from the same retrieved facts.

4. Run the app:

```bash
python -m src.main
```

Console output is also written to `logs/app.log`.

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



