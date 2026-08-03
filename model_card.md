# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**TrueVibe 1.0**

---

## 2. Intended Use

This recommender suggests songs from a small catalog based on a user's
taste. A user can type a plain sentence (like "I want calm music for
studying") or answer a few short questions about genre, mood, energy, and
acoustic preference. It assumes taste is simple and doesn't change during
one run. This is a classroom project, not a real production app.

---

## 3. How the Model Works

Each song has a genre, mood, energy level, and acousticness. The model
compares those to what the user asked for. It gives points for each match:
- matching genre, matching mood, and energy close to the user's target. 
- It also gives a small bonus if the acousticness fits the user's preference.
- Songs with more points are ranked higher.

Changes from the starter logic:
- Filled in all the scoring and ranking, which were empty before.
- Added a step that finds similar songs in the catalog for context.
- Added an AI step that writes a short explanation, using only real facts about the song (no made-up details).
- Added a way to type a plain sentence, which AI turns into the same
  genre/mood/energy preferences used for scoring.

---

## 4. Data

The catalog has 10 songs in `data/songs.csv`. Genres: pop, lofi, rock,
ambient, jazz, synthwave, indie pop. Moods: happy, chill, intense, moody, relaxed, focused. I didn't add or remove any songs — I used the starter data as-is.

What's missing: no lyrics, no real listening history, and no genres like hip-hop, metal, or classical. With only 10 songs, most tastes aren't actually represented.

---

## 5. Strengths

- Works well when a user's taste exactly matches a catalog genre and mood (for example: pop + happy).
- Energy matching works smoothly since it's just a number, not an exact string match.
- Explanations always stick to real facts about the song — they never
  invent an artist, genre, or stat that isn't in the data.

---

## 6. Limitations and Bias

- Small catalog: only 10 songs, so it can only reorder what's already
  there.
- Genre matching is exact-string only. "pop" gets no credit from "indie pop," even though they're close.
- The scoring weights (genre worth more than mood, mood worth more than acousticness) are my own guesses, not based on real data.
- No memory between runs — it can't learn from what a user skipped or
  liked before.

---

## 7. Evaluation

I tested it by hand with a few different profiles: pop/happy, lofi/chill with acoustic preference, rock/intense, and the plain sentence "I want calm music for studying." I checked each explanation against the real CSV data to make sure nothing was invented. I also wrote 12 automated tests (`pytest`), covering scoring, similar-song lookup, and what happens when the AI call fails or there's no API key — all 12 pass.

What surprised me: almost every failure I hit was not a logic bug. It was billing, API quota limits, or a local certificate issue. The actual scoring and explanation logic worked correctly once a real connection to the model was made.

---

## 8. Future Work

- Use a bigger, more diverse song catalog.
- Match similar genres (like "pop" and "indie pop") instead of requiring an exact match.
- Remember a user's choices across runs instead of starting fresh each
  time.
- Let a user say what they *didn't* like, and adjust future
  recommendations.

---

## 9. Personal Reflection

Building this taught me that a recommender is really just a scoring rule someone picked by hand — and that rule is exactly where bias can sneak in. I was surprised by how well "grounding" worked: giving the AI only real facts to work with stopped it from making things up, every time I checked. It changed how I think about apps like Spotify — their "recommended for you" picks are probably the same idea as mine, just with a much bigger catalog and much more data behind the weights.

---

## 10. Responsible AI Reflection

### What are the limitations or biases in your system?

- Small catalog: only 10 songs, so it can only reorder what's already
  there.
- Genre matching is exact-string only. "pop" gets no credit from "indie pop," even though they're similar.
- The scoring weights (genre +2.0, mood +1.5, energy, acousticness) are my own guesses, not based on real data.
- No memory between runs. It can't learn from what a user skipped or
  liked.

### Could your AI be misused, and how would you prevent that?

- Main risk: prompt injection. The free-text input goes straight into a prompt sent to Gemini.
- Someone could type something trying to trick the model into returning bad data.
- Prevention already in the code: `_sanitize_parsed_prefs()` only accepts genre/mood values that already exist in the catalog, and clamps energy to 0.0–1.0. Bad output can't reach the scoring engine.
- If this were a public app, I'd also add input length limits and rate
  limiting.

### What surprised you while testing your AI's reliability?

- Most failures weren't AI bugs — they were billing, API quota, and a
  local TLS certificate issue.
- The grounded prompt itself worked correctly the first time it reached a real model.
- The real Gemini call understood "calm music for studying" better than my own keyword fallback did.

### Collaboration with AI

- **Helpful suggestion:** Claude designed the explanations to retrieve
  real facts first, then generate text using only those facts. This kept every explanation grounded — no invented songs or attributes.
- **Flawed suggestion:** Claude's first default model choice,
  `gemini-2.0-flash`, didn't actually work with my account (zero free-tier quota). We had to test a few models before finding one that worked, `gemini-flash-lite-latest`.
