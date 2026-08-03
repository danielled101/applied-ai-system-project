"""
Command line runner for the Music Recommender Simulation.
"""

import logging
import os

from src.recommender import load_songs, recommend_songs, parse_user_request

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = {"genre": "pop", "mood": "happy", "energy": 0.8}


def configure_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/app.log")],
    )


def prompt_user_profile(songs) -> dict:
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})
    print(f"Available genres: {', '.join(genres)}")
    print(f"Available moods:  {', '.join(moods)}\n")

    genre = input("Favorite genre: ").strip().lower()
    mood = input("Favorite mood: ").strip().lower()

    energy = 0.5
    energy_raw = input("Target energy, 0.0 (calm) to 1.0 (intense): ").strip()
    try:
        energy = max(0.0, min(1.0, float(energy_raw)))
    except ValueError:
        print(f"Couldn't parse '{energy_raw}' as a number, defaulting energy to 0.5.")

    acoustic_raw = input("Do you like acoustic sound? (y/n): ").strip().lower()
    likes_acoustic = acoustic_raw.startswith("y")

    return {"genre": genre, "mood": mood, "energy": energy, "likes_acoustic": likes_acoustic}


def main() -> None:
    configure_logging()
    songs = load_songs("data/songs.csv")

    try:
        request_text = input(
            "Describe what you're in the mood for (e.g. 'I want calm music for "
            "studying'), or press Enter to answer a few questions instead: "
        ).strip()
        if request_text:
            user_prefs = parse_user_request(request_text, songs)
            print(f"\nGot it - interpreting that as: {user_prefs}\n")
        else:
            user_prefs = prompt_user_profile(songs)
    except EOFError:
        logger.warning("No interactive input available; using the default demo profile")
        user_prefs = DEFAULT_PROFILE

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
