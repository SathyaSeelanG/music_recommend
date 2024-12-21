# import os
# import cv2
# from flask import Flask, render_template, request, redirect, url_for
# from fer import FER
# from spotipy import Spotify
# from spotipy.oauth2 import SpotifyClientCredentials
# from youtubesearchpython import VideosSearch
# from dotenv import load_dotenv

# load_dotenv()

# app = Flask(__name__)

# # Spotify API credentials (set these as environment variables)
# SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
# SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# # Initialize Spotify client
# spotify = Spotify(auth_manager=SpotifyClientCredentials(
#     client_id=SPOTIFY_CLIENT_ID,
#     client_secret=SPOTIFY_CLIENT_SECRET
# ))

# def capture_mood():
#     """Capture face and detect mood using OpenCV and FER."""
#     cap = cv2.VideoCapture(0)
#     detector = FER(mtcnn=True)

#     mood_detected = "neutral"  # Default mood
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         cv2.imshow("Press 'q' to capture your mood", frame)
#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q'):  # Press 'q' to capture the mood
#             mood = detector.detect_emotions(frame)
#             if mood:
#                 mood_detected = max(mood[0]["emotions"], key=mood[0]["emotions"].get)
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return mood_detected

# def get_music_recommendations(mood, language="English"):
#     """Fetch music recommendations based on mood."""
#     mood_playlists = {
#         "happy": "Happy Hits",
#         "sad": "Sad Vibes",
#         "angry": "Pump It Up",
#         "neutral": "Chill Hits",
#         "surprise": "Party Time"
#     }
#     playlist_name = mood_playlists.get(mood, "Top Hits")
    
#     # Spotify search
#     results = spotify.search(q=f"{playlist_name} {language}", type="playlist", limit=1)
#     if results["playlists"]["items"]:
#         playlist_id = results["playlists"]["items"][0]["id"]
#         tracks = spotify.playlist_tracks(playlist_id)
#         return [track["track"]["name"] + " - " + track["track"]["artists"][0]["name"] for track in tracks["items"]]
#     else:
#         return []

# @app.route("/", methods=["GET", "POST"])
# def index():
#     """Home page."""
#     if request.method == "POST":
#         language = request.form.get("language")
#         return redirect(url_for("detect_mood", language=language))
#     return render_template("index.html")

# @app.route("/detect_mood/<language>", methods=["GET"])
# def detect_mood(language):
#     """Detect mood using webcam."""
#     mood = capture_mood()
#     return redirect(url_for("recommendations", mood=mood, language=language))

# @app.route("/recommendations", methods=["GET"])
# def recommendations():
#     """Display music recommendations."""
#     mood = request.args.get("mood", "neutral")
#     language = request.args.get("language", "English")
#     songs = get_music_recommendations(mood, language)
#     return render_template("recommendations.html", mood=mood, language=language, songs=songs)

# if __name__ == "__main__":
#     app.run(debug=True)
import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
from fer import FER
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Initialize Spotify client
spotify = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

def capture_mood():
    """Capture face and detect mood using OpenCV and FER."""
    cap = cv2.VideoCapture(0)
    detector = FER(mtcnn=True)

    mood_detected = "neutral"  # Default mood
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Press 'q' to capture your mood", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):  # Press 'q' to capture the mood
            mood = detector.detect_emotions(frame)
            if mood:
                mood_detected = max(mood[0]["emotions"], key=mood[0]["emotions"].get)
            break

    cap.release()
    cv2.destroyAllWindows()
    return mood_detected

# def get_music_recommendations(mood, language="English"):
#     """Fetch music recommendations based on mood."""
#     mood_playlists = {
#         "happy": "Happy Hits",
#         "sad": "Sad Vibes",
#         "angry": "Pump It Up",
#         "neutral": "Chill Hits",
#         "surprise": "Party Time"
#     }
#     playlist_name = mood_playlists.get(mood, "Top Hits")
    
#     # Spotify search
#     results = spotify.search(q=f"{playlist_name} {language}", type="playlist", limit=1)
#     if results["playlists"]["items"]:
#         playlist_id = results["playlists"]["items"][0]["id"]
#         tracks = spotify.playlist_tracks(playlist_id)
#         return [track["track"]["name"] + " - " + track["track"]["artists"][0]["name"] for track in tracks["items"]]
#     else:
#         return []
#new method for youtube music recommendation
def get_music_recommendations(mood, language="English"):
    """Fetch music recommendations based on mood."""
    mood_playlists = {
        "happy": "Happy Hits",
        "sad": "Sad Vibes",
        "angry": "Pump It Up",
        "neutral": "Chill Hits",
        "surprise": "Party Time"
    }
    playlist_name = mood_playlists.get(mood, "Top Hits")
    
    # Attempt to fetch from Spotify
    try:
        results = spotify.search(q=f"{playlist_name} {language}", type="playlist", limit=1)
        if results["playlists"]["items"]:
            playlist_id = results["playlists"]["items"][0]["id"]
            tracks = spotify.playlist_tracks(playlist_id)
            return [track["track"]["name"] + " - " + track["track"]["artists"][0]["name"] for track in tracks["items"]]
        else:
            raise Exception("No playlists found on Spotify.")
    except Exception as e:
        print(f"Spotify API error: {e}")
        # Fallback to YouTube
        videos_search = VideosSearch(f"{playlist_name} {language}", limit=5)
        response = videos_search.result()
        return [video['title'] for video in response['videos']]

@app.route("/", methods=["GET", "POST"])
def index():
    """Home page."""
    if request.method == "POST":
        language = request.form.get("language")
        return redirect(url_for("detect_mood", language=language))
    return render_template("index.html")

@app.route("/detect_mood/<language>", methods=["GET"])
def detect_mood(language):
    """Detect mood using webcam."""
    mood = capture_mood()
    return redirect(url_for("recommendations", mood=mood, language=language))

@app.route("/recommendations", methods=["GET"])
def recommendations():
    """Display music recommendations."""
    mood = request.args.get("mood", "neutral")
    language = request.args.get("language", "English")
    songs = get_music_recommendations(mood, language)
    return render_template("recommendations.html", mood=mood, language=language, songs=songs)

if __name__ == "__main__":
    app.run(debug=True)
