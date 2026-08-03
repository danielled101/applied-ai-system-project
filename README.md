# 🎵 Music Recommender Simulation

## Original Project (Modules 1–3)

This repository is an evolution of **Music Recommender Simulation**, my
Module 1–3 project for CodePath's AI110 course (based on the course's
`ai110-module3show-musicrecommendersimulation-starter` starter kit). The
original project's goal was to represent songs and a user "taste profile" as
plain data, design a hand-written scoring rule that turns that data into
ranked recommendations, and evaluate/reflect on where a simple rule-based
recommender gets things right or wrong compared to real-world AI
recommenders. This new repository keeps that original scoring engine intact
and builds an advanced AI feature on top of it (below), so the Module 1–3
work can be compared side-by-side with the extended system.

---

## Summary

A small music recommender that scores a 10-song catalog against a user's
taste profile, ranks the results, and — instead of a generic label — explains
*why* each song was picked using **Retrieval-Augmented Generation (RAG)**.
It matters because it's a miniature, inspectable version of the same
pattern real recommender systems (Spotify, Netflix, etc.) rely on: score →
rank → justify, with the "justify" step grounded in retrieved facts rather
than an LLM guessing, so it's reproducible for a grader with or without
network access to an LLM.

---

## Architecture Overview

See [`diagrams/architecture.mmd`](diagrams/architecture.mmd) for the full
Mermaid diagram. In short, data flows through six stages:

1. **Data Layer** — `load_songs()` reads `data/songs.csv` into records.
2. **NL Parser** — `parse_user_request()` turns a free-text request like
   *"I want calm music for studying"* into structured preferences, grounded
   in the catalog's actual genre/mood vocabulary so it can't invent a value
   that will never match anything. Falls back to deterministic
   keyword/synonym matching if no API key or the call fails.
3. **Scoring Engine** — `score_song()` compares each song to those
   preferences (genre, mood, energy, acousticness) and returns both a
   numeric score and the specific reasons it matched.
4. **Retriever** — `retrieve_similar_songs()` finds each recommended song's
   nearest neighbors in the catalog by attribute distance (energy, valence,
   danceability, acousticness, tempo) — the "R" in RAG.
5. **Generation Agent** — `generate_explanation()` sends the retrieved
   reasons + neighbors to Gemini, instructed to explain the recommendation
   using *only* that context.
6. **Guardrail / Evaluator** — checks the API key exists, the call
   succeeded, and the response is non-empty; on any failure it falls back to
   a deterministic template built from the same retrieved facts.

The diagram also shows the **testing layer**: the `pytest` suite
(`tests/test_recommender.py`) validates the scoring and explanation logic
automatically, and a developer-review step (running the CLI, reading
`logs/app.log`) is where a human checks explanation quality by hand.

---

## Setup Instructions

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional, for AI-generated explanations)* Create a `.env` file in the
   project root with a free Gemini API key from
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey):

   ```
   GEMINI_API_KEY=your-key-here
   GEMINI_MODEL=gemini-flash-lite-latest
   ```

   Without a key, the app still runs end-to-end — explanations fall back to
   a deterministic template built from the same retrieved facts, so setup
   never blocks on getting a key.

4. Run the app:

   ```bash
   python -m src.main
   ```

   It will ask you to describe what you're in the mood for — you can type a
   free-text request like `I want calm music for studying`, or press Enter
   to answer a few guided questions (genre, mood, target energy, acoustic
   preference) instead. Either way it prints personalized ranked
   recommendations, logged to `logs/app.log`. If run without an interactive
   terminal (e.g. in a CI pipeline), it safely falls back to a fixed demo
   profile instead of crashing.

5. Run the tests:

   ```bash
   pytest
   ```

   The tests exercise the fallback path (no network calls), so they run the
   same way for every grader regardless of whether a key is configured.

---

## Sample Interactions

These are real outputs from the live system (Gemini-generated, using the
retrieved scoring reasons and similar-song neighbors as grounding — not
hand-written examples):

**Natural-language input:** `"I want calm music for studying."`
```
Parsed preferences: {'mood': 'focused', 'energy': 0.2}

Focus Flow by LoRoom (score 2.30)
  We think you'll love "Focus Flow" by LoRoom because its focused mood
  matches your preferred mood. Enjoy this lofi track as you listen!

Spacewalk Thoughts by Orbit Bloom (score 0.92)
  We think you'll enjoy "Spacewalk Thoughts" by Orbit Bloom! Its energy
  level of 0.28 is a great match for your target energy of 0.2.
```
Note the request never contains the words "focused," "lofi," or a number —
Gemini inferred all of that from context, constrained to the catalog's real
mood vocabulary by the NL parser's grounding step.

**Structured input:** `{genre: pop, mood: happy, energy: 0.8}`
```
Sunrise City by Neon Echo (score 4.48)
  We recommended "Sunrise City" by Neon Echo because its pop genre matches your favorite genre and its happy mood fits your preferred mood. Plus, its energy level of 0.82 is right on target with your preferred energy of 0.80!

Gym Hero by Max Pulse (score 2.87)
  We think you'll love "Gym Hero" by Max Pulse because its pop genre
  matches your favorite genre! Plus, its high energy level of 0.93 is very close to your target energy of 0.80.
```

**Structured input:** `{genre: lofi, mood: chill, energy: 0.3, likes_acoustic: true}`
```
Library Rain by Paper Lanterns (score 5.20)
  We recommend "Library Rain" by Paper Lanterns because its lofi genre and chill mood match your favorites. It also features a high acousticness that fits your taste, and an energy level close to your target!

Midnight Coding by LoRoom (score 5.13)
  We recommended "Midnight Coding" by LoRoom because its lofi genre and chill mood match your favorites. Plus, its energy and high acousticness fit your preferred sound!
```

**Structured input:** `{genre: rock, mood: intense, energy: 0.9}`
```
Storm Runner by Voltline (score 4.49)
  We recommended "Storm Runner" by Voltline because its rock genre matches your favorite, and its intense mood fits your preferred vibe. Plus, its high energy level of 0.91 is right on target with your 0.90 preference!

Gym Hero by Max Pulse (score 2.47)
  We recommended "Gym Hero" by Max Pulse because its intense mood matches your preferred mood. Additionally, its energy level of 0.93 is very close to your target energy of 0.90!
```

Note every explanation only references attributes that are actually in the
catalog data — no invented artists, genres, or stats.

---

## Design Decisions

- **RAG over a bolt-on script.** The advanced AI feature had to change how
  the system *actually* behaves, not just print alongside it. Retrieval
  happens inside `recommend_songs()`/`explain_recommendation()` on every
  call — there's no separate "AI mode" to opt into.
- **Grounding before generation.** Rather than letting the model free-write
  an explanation, I retrieve the specific scoring reasons and nearest-
  neighbor songs first and instruct the model to use *only* those facts.
  Trade-off: explanations are less flowery/creative than an ungrounded
  prompt would produce, but they can't hallucinate a genre or artist that
  isn't in the catalog.
- **Deterministic fallback, always.** Missing key, missing package, network
  failure, or an empty response all degrade to the same template built from
  the retrieved facts, logged clearly. Trade-off: a grader without a key
  never sees the LLM-generated prose, only the template — but the system
  never crashes and stays reproducible, which mattered more than always
  showing the flashiest output.
- **Gemini instead of Claude.** I originally built this against the
  Anthropic API, but my account had no billing credit. Gemini's free tier
  let me ship a fully working, fully free version — the retrieval/guardrail
  architecture is provider-agnostic, so swapping `generate_explanation()`'s
  backend didn't require touching the rest of the system.
- **Attribute-distance retrieval, not embeddings.** With a 10-song catalog,
  a full embedding/vector-search pipeline would be over-engineering.
  Euclidean distance over five numeric audio attributes is simple, fast,
  and easy to unit test; the trade-off is it wouldn't scale gracefully to a
  catalog with thousands of songs without revisiting this choice.
- **Natural-language input, grounded the same way as explanations.** Typing
  a free-text mood description is a much more natural entry point than
  answering four separate questions, but a naive prompt could ask Gemini to
  return *any* genre/mood string, including ones that don't exist in the
  catalog and would silently fail to match anything in `score_song()`. The
  parser instead gives Gemini the catalog's actual genre/mood list and
  requires it to choose from that list (or return null). Trade-off: the
  keyword/synonym fallback (for when there's no key) is necessarily cruder
  than the LLM at understanding phrasing it wasn't explicitly given a
  synonym for.

---

## Testing Summary

### Automated tests

**12 out of 12 automated tests passed** (`pytest -v`). Beyond the two
starter tests, the suite specifically targets reliability, not just
happy-path scoring:

| Test | What it proves |
|---|---|
| `test_score_song_gives_reasons_for_each_match` | scoring produces the matched-preference reasons the RAG step depends on |
| `test_score_song_no_match_still_returns_a_reason` | never returns an empty reasons list, even with zero matches |
| `test_retrieve_similar_songs_excludes_target_and_respects_top_n` | retrieval never returns the song being explained as its own neighbor |
| `test_recommend_songs_returns_k_results_sorted_by_score` | ranking is correctly sorted, not just filtered |
| `test_recommend_songs_handles_empty_catalog_without_crashing` | empty input degrades gracefully instead of throwing |
| `test_generate_explanation_falls_back_without_api_key` | the guardrail engages when `GEMINI_API_KEY` is unset |
| `test_generate_explanation_falls_back_when_api_call_fails` | the guardrail engages when the API call raises (simulated outage), not just when the key is missing |
| `test_fallback_parse_request_maps_synonyms_not_in_catalog_vocabulary` | the NL keyword fallback maps everyday words ("calm," "studying") to real catalog moods, not just literal matches |
| `test_fallback_parse_request_only_ever_returns_catalog_values` | the NL fallback parser can never emit a genre/mood the catalog doesn't have |
| `test_parse_user_request_falls_back_without_api_key` | the NL parser's guardrail engages when `GEMINI_API_KEY` is unset |

That last test is the important one for reliability: it mocks `google.genai.Client`
to raise on every call and asserts the system still returns a valid, non-empty
explanation instead of crashing — proof the guardrail works even when Gemini
itself is down, not just when a key was never configured.

### Human evaluation (groundedness check)

Automated tests can check that *something* came back, but not whether an
LLM-generated explanation is telling the truth about the catalog. I manually
checked each real Gemini output (the four samples above) against the actual
CSV data:

| Test Input | Evaluation Criteria | Result |
|---|---|---|
| `{genre: pop, mood: happy, energy: 0.8}` | Only cites attributes present in `songs.csv` for the recommended song; no invented artist/genre | Pass |
| `{genre: lofi, mood: chill, energy: 0.3, likes_acoustic: true}` | Only cites attributes present in `songs.csv`; correctly reflects `likes_acoustic` in the reasoning | Pass |
| `{genre: rock, mood: intense, energy: 0.9}` | Only cites attributes present in `songs.csv`; energy comparison numbers match the source data | Pass |
| `"I want calm music for studying."` | NL parser maps to a real catalog mood/energy (not invented); recommended song is genuinely a lofi/study-style track | Pass |
| No `GEMINI_API_KEY` set | App runs to completion, returns a non-empty, sensible explanation via the template | Pass |
| Empty song catalog (`[]`) | Returns `[]` instead of raising | Pass |

**What didn't work initially:**
- My first Anthropic API key had no billing credit, so every call failed — looked like a connection error at first, but tracing it down showed it was a billing/credit issue, not a code bug.
- The starter's `src/main.py` imported `recommender` as a bare top-level
  module, which breaks under the README's own documented invocation
  (`python -m src.main`). Fixed to `from src.recommender import ...`.
- My Gemini key's free tier returned `429 RESOURCE_EXHAUSTED` (quota limit
  0) for `gemini-2.0-flash`; listing available models and testing a few
  showed `gemini-flash-lite-latest` had quota available, so that became the
  default model.

**What I learned:** most of the "AI is broken" symptoms I hit during this project were actually infrastructure problems (billing, quotas, TLS/certificates) rather than logic bugs — the lesson was to isolate each layer (network reachability, auth, quota, then app logic) instead of assuming the first error message is the root cause.

---

## Reflection

Read [`model_card.md`](model_card.md) for the graded responsible-AI
reflection — how I collaborated with AI on this project, one AI suggestion
that helped and one that was flawed, and this system's limitations.

Beyond that: building the retrieval step made concrete something I'd only
understood abstractly before — that "grounding" isn't a checkbox, it's a
design constraint that shapes the prompt, the guardrails, and the fallback
path all at once. It also reinforced that a recommender's scoring rule
*is* its opinion about what matters (genre match weighted more than
acousticness, here), and that opinion is exactly where bias would creep in
if this were a real product instead of a 10-song classroom catalog.
