import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "id", "title", "artist", "genre", "mood", "energy",
    "tempo_bpm", "valence", "danceability", "acousticness",
}

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        prefs = _profile_to_prefs(user)
        scored = [(song, score_song(prefs, asdict(song))[0]) for song in self.songs]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        prefs = _profile_to_prefs(user)
        song_dict = asdict(song)
        _, reasons = score_song(prefs, song_dict)
        catalog = [asdict(s) for s in self.songs]
        similar = retrieve_similar_songs(song_dict, catalog, top_n=3)
        return generate_explanation(prefs, song_dict, reasons, similar)


def _profile_to_prefs(user: UserProfile) -> Dict:
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    logger.info("Loading songs from %s", csv_path)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error("Songs catalog not found at %s", csv_path)
        raise

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"songs.csv is missing required columns: {sorted(missing)}")

    songs = df.to_dict(orient="records")
    logger.info("Loaded %d songs", len(songs))
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    reasons: List[str] = []
    score = 0.0

    genre_pref = user_prefs.get("genre")
    mood_pref = user_prefs.get("mood")
    energy_pref = user_prefs.get("energy")
    likes_acoustic = user_prefs.get("likes_acoustic")

    if genre_pref and song.get("genre") == genre_pref:
        score += 2.0
        reasons.append(f"its genre ({song.get('genre')}) matches your favorite genre")

    if mood_pref and song.get("mood") == mood_pref:
        score += 1.5
        reasons.append(f"its mood ({song.get('mood')}) matches your preferred mood")

    if energy_pref is not None:
        energy_diff = abs(float(song.get("energy", 0.0)) - float(energy_pref))
        score += max(0.0, 1.0 - energy_diff)
        if energy_diff < 0.15:
            reasons.append(
                f"its energy ({float(song.get('energy', 0.0)):.2f}) is close to your target energy ({float(energy_pref):.2f})"
            )

    if likes_acoustic is not None:
        acousticness = float(song.get("acousticness", 0.0))
        if likes_acoustic and acousticness >= 0.6:
            score += 0.75
            reasons.append(f"its high acousticness ({acousticness:.2f}) fits your taste for acoustic sound")
        elif not likes_acoustic and acousticness < 0.4:
            score += 0.25
            reasons.append("its produced, low-acoustic sound fits your preference")

    if not reasons:
        reasons.append("it is the closest overall match available in the catalog")

    return round(score, 4), reasons


def retrieve_similar_songs(target_song: Dict, catalog: List[Dict], top_n: int = 3) -> List[Dict]:
    """
    Retrieval step: finds the catalog songs whose attribute profile is
    closest to target_song, to ground the explanation in real neighbors
    instead of letting the model invent comparisons.
    """
    fields = ["energy", "valence", "danceability", "acousticness"]

    def vector(song: Dict) -> List[float]:
        return [float(song.get(f, 0.0)) for f in fields] + [float(song.get("tempo_bpm", 0.0)) / 200.0]

    target_vec = vector(target_song)
    scored = []
    for song in catalog:
        if song.get("id") == target_song.get("id"):
            continue
        distance = sum((a - b) ** 2 for a, b in zip(target_vec, vector(song))) ** 0.5
        scored.append((distance, song))

    scored.sort(key=lambda pair: pair[0])
    neighbors = [song for _, song in scored[:top_n]]
    logger.debug(
        "Retrieved %d similar songs for '%s': %s",
        len(neighbors), target_song.get("title"), [s.get("title") for s in neighbors],
    )
    return neighbors


def _fallback_explanation(song: Dict, reasons: List[str], similar_songs: List[Dict]) -> str:
    reason_text = "; ".join(reasons)
    explanation = f'We recommended "{song.get("title")}" by {song.get("artist")} because {reason_text}.'
    if similar_songs:
        names = ", ".join(f'"{s.get("title")}" by {s.get("artist")}' for s in similar_songs[:2])
        explanation += f" It also has a similar sound profile to {names} already in the catalog."
    return explanation


def generate_explanation(
    user_prefs: Dict,
    song: Dict,
    reasons: List[str],
    similar_songs: List[Dict],
) -> str:
    """
    RAG step: has Gemini write the explanation grounded in the retrieved
    facts (score reasons + similar-song neighbors) rather than free-form.
    Falls back to a deterministic template if the API key is missing,
    the package isn't installed, or the call errors out, so this never
    breaks a recommendation run.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set; using template explanation instead of RAG generation")
        return _fallback_explanation(song, reasons, similar_songs)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("google-genai package not installed; using template explanation instead of RAG generation")
        return _fallback_explanation(song, reasons, similar_songs)

    context = {
        "recommended_song": {k: song.get(k) for k in ("title", "artist", "genre", "mood", "energy", "acousticness")},
        "matched_reasons": reasons,
        "similar_catalog_songs": [
            {k: s.get(k) for k in ("title", "artist", "genre", "mood")} for s in similar_songs
        ],
        "user_preferences": user_prefs,
    }

    prompt = (
        "You are writing a short explanation for why a song was recommended to a user. "
        "Use ONLY the facts provided in RETRIEVED_CONTEXT below. Do not invent song attributes, "
        "artists, or facts that are not present in the context. Keep it to 1-3 friendly sentences.\n\n"
        f"RETRIEVED_CONTEXT:\n{context}"
    )

    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=15000))
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=200, temperature=0.4),
            )
            text = (response.text or "").strip()
            if text:
                logger.info("Generated RAG explanation for '%s' via %s", song.get("title"), DEFAULT_MODEL)
                return text
            last_error = ValueError("empty response from model")
        except Exception as exc:  # guardrail: any API failure must degrade, never crash a recommendation
            last_error = exc
            logger.warning("Gemini API call failed (attempt %d/2): %s", attempt + 1, exc)

    logger.error("Falling back to template explanation after API failure: %s", last_error)
    return _fallback_explanation(song, reasons, similar_songs)


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    if not songs:
        logger.warning("recommend_songs called with an empty catalog")
        return []

    scored = [(song, *score_song(user_prefs, song)) for song in songs]
    scored.sort(key=lambda item: item[1], reverse=True)

    results = []
    for song, score, reasons in scored[:k]:
        similar = retrieve_similar_songs(song, songs, top_n=3)
        explanation = generate_explanation(user_prefs, song, reasons, similar)
        results.append((song, score, explanation))

    return results
