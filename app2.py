import os
import cv2
from fer import FER
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from youtube_search import YoutubeSearch

# Initialize Spotify client
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
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

def get_spotify_recommendations(mood):
    """Fetch music recommendations from Spotify based on mood."""
    mood_playlists = {
        "happy": "Happy Hits",
        "sad": "Sad Vibes",
        "angry": "Pump It Up",
        "neutral": "Chill Hits",
        "surprise": "Party Time"
    }
    playlist_name = mood_playlists.get(mood, "Top Hits")
    
    results = spotify.search(q=f"{playlist_name}", type="playlist", limit=1)
    if results["playlists"]["items"]:
        playlist_id = results["playlists"]["items"][0]["id"]
        tracks = spotify.playlist_tracks(playlist_id)
        return [track["track"]["name"] + " - " + track["track"]["artists"][0]["name"] for track in tracks["items"]]
    else:
        return []

def get_youtube_recommendations(mood):
    """Fetch music recommendations from YouTube based on mood."""
    mood_playlists = {
        "happy": "Happy Hits",
        "sad": "Sad Vibes",
        "angry": "Pump It Up",
        "neutral": "Chill Hits",
        "surprise": "Party Time"
    }
    playlist_name = mood_playlists.get(mood, "Top Hits")
    
    results = YoutubeSearch(f"{playlist_name} playlist", max_results=5).to_dict()
    return [video['title'] for video in results]

if __name__ == "__main__":
    mood = capture_mood()
    print(f"Detected mood: {mood}")
    
    spotify_songs = get_spotify_recommendations(mood)
    youtube_videos = get_youtube_recommendations(mood)
    
    print("\nSpotify Recommendations:")
    for song in spotify_songs:
        print(song)
    
    print("\nYouTube Recommendations:")
    for video in youtube_videos:
        print(video)
