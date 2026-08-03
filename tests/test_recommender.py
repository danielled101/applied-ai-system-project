import pytest

from src.recommender import (
    Song,
    UserProfile,
    Recommender,
    score_song,
    retrieve_similar_songs,
    recommend_songs,
    generate_explanation,
    parse_user_request,
    _fallback_parse_request,
)

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# --- Reliability tests: scoring, retrieval, and guardrail/fallback behavior ---

CATALOG = [
    {
        "id": 1, "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop",
        "mood": "happy", "energy": 0.82, "tempo_bpm": 118, "valence": 0.84,
        "danceability": 0.79, "acousticness": 0.18,
    },
    {
        "id": 2, "title": "Midnight Coding", "artist": "LoRoom", "genre": "lofi",
        "mood": "chill", "energy": 0.42, "tempo_bpm": 78, "valence": 0.56,
        "danceability": 0.62, "acousticness": 0.71,
    },
    {
        "id": 3, "title": "Storm Runner", "artist": "Voltline", "genre": "rock",
        "mood": "intense", "energy": 0.91, "tempo_bpm": 152, "valence": 0.48,
        "danceability": 0.66, "acousticness": 0.10,
    },
    {
        "id": 4, "title": "Coffee Shop Stories", "artist": "Slow Stereo", "genre": "jazz",
        "mood": "relaxed", "energy": 0.37, "tempo_bpm": 90, "valence": 0.71,
        "danceability": 0.54, "acousticness": 0.89,
    },
]


def test_score_song_gives_reasons_for_each_match():
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    score, reasons = score_song(prefs, CATALOG[0])

    assert score > 0
    assert any("genre" in r for r in reasons)
    assert any("mood" in r for r in reasons)


def test_score_song_no_match_still_returns_a_reason():
    prefs = {"genre": "jazz", "mood": "sad", "energy": 0.1}
    score, reasons = score_song(prefs, CATALOG[0])

    assert len(reasons) >= 1
    assert isinstance(score, float)


def test_retrieve_similar_songs_excludes_target_and_respects_top_n():
    neighbors = retrieve_similar_songs(CATALOG[0], CATALOG, top_n=2)

    assert len(neighbors) == 2
    assert all(song["id"] != CATALOG[0]["id"] for song in neighbors)


def test_recommend_songs_returns_k_results_sorted_by_score():
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    results = recommend_songs(prefs, CATALOG, k=2)

    assert len(results) == 2
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0][0]["genre"] == "pop"


def test_recommend_songs_handles_empty_catalog_without_crashing():
    results = recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.8}, [], k=5)
    assert results == []


def test_generate_explanation_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    explanation = generate_explanation(
        {"genre": "pop", "mood": "happy", "energy": 0.8},
        CATALOG[0],
        ["its genre (pop) matches your favorite genre"],
        [CATALOG[1]],
    )

    assert explanation.startswith('We recommended "Sunrise City"')


def test_fallback_parse_request_maps_synonyms_not_in_catalog_vocabulary():
    genres = sorted({s["genre"] for s in CATALOG})
    moods = sorted({s["mood"] for s in CATALOG})

    prefs = _fallback_parse_request("I want calm music for studying.", genres, moods)

    # "calm"/"studying" never literally appear in the mood vocabulary
    # (happy/chill/intense/relaxed) - this proves the synonym map, not a
    # fluke substring match, is what produced the result.
    assert prefs["mood"] == "relaxed"
    assert prefs["energy"] < 0.5


def test_fallback_parse_request_only_ever_returns_catalog_values():
    genres = sorted({s["genre"] for s in CATALOG})
    moods = sorted({s["mood"] for s in CATALOG})

    prefs = _fallback_parse_request("surprise me with something totally random", genres, moods)

    if "genre" in prefs:
        assert prefs["genre"] in genres
    if "mood" in prefs:
        assert prefs["mood"] in moods


def test_parse_user_request_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    prefs = parse_user_request("I want calm music for studying.", CATALOG)

    assert prefs["mood"] == "relaxed"
    assert prefs["energy"] < 0.5


def test_generate_explanation_falls_back_when_api_call_fails(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    class ExplodingClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated API outage")

    monkeypatch.setattr("google.genai.Client", ExplodingClient)

    explanation = generate_explanation(
        {"genre": "pop", "mood": "happy", "energy": 0.8},
        CATALOG[0],
        ["its genre (pop) matches your favorite genre"],
        [CATALOG[1]],
    )

    assert explanation.startswith('We recommended "Sunrise City"')
