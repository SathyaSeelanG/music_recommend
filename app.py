import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
from fer import FER
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from youtubesearchpython import VideosSearch
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
    
    # Set window properties
    cv2.namedWindow("Capturing your mood...", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Capturing your mood...", 0, 0)  # Move window to front
    cv2.setWindowProperty("Capturing your mood...", cv2.WND_PROP_TOPMOST, 1)  # Keep window on top
    
    mood_detected = "neutral"  # Default mood
    countdown = 3  # Countdown in seconds
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate remaining time
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        remaining_time = max(0, countdown - int(elapsed_time))
        
        # Add countdown text to frame
        if remaining_time > 0:
            text = f"Capturing in {remaining_time}..."
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow("Capturing your mood...", frame)
        
        # Break if countdown is done
        if elapsed_time >= countdown:
            # Detect mood
            mood = detector.detect_emotions(frame)
            if mood:
                mood_detected = max(mood[0]["emotions"], key=mood[0]["emotions"].get)
            break
        
        # Allow manual exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Force close any remaining windows
    for i in range(4):
        cv2.waitKey(1)
    
    return mood_detected

def get_music_recommendations(mood, language="English"):
    """Fetch music recommendations based on mood."""
    mood_playlists = {
        "happy": "Happy Hits",
        "sad": "Sad Songs",
        "angry": "Aggressive Songs",
        "neutral": "Relaxing Music",
        "surprise": "Party Hits",
        "fear": "Calming Music",
        "disgust": "Feel Good Music"
    }
    
    playlist_name = mood_playlists.get(mood, "Popular Music")
    
    # Attempt to fetch from Spotify
    try:
        # Try multiple search terms for better results
        search_terms = [
            f"{playlist_name} {language}",
            f"Popular {language} {mood} songs",
            f"{language} {mood} playlist"
        ]
        
        for search_term in search_terms:
            results = spotify.search(q=search_term, type="playlist", limit=1)
            if results["playlists"]["items"]:
                playlist_id = results["playlists"]["items"][0]["id"]
                tracks = spotify.playlist_tracks(playlist_id)
                song_info = []
                for track in tracks["items"][:10]:  # Limit to 10 songs
                    if track["track"] is None:
                        continue
                    song_name = track["track"]["name"]
                    artist_name = track["track"]["artists"][0]["name"]
                    album_image_url = track["track"]["album"]["images"][0]["url"] if track["track"]["album"]["images"] else "https://via.placeholder.com/300"
                    spotify_url = track["track"]["external_urls"].get("spotify", "")
                    song_info.append({
                        "song": f"{song_name} - {artist_name}",
                        "image_url": album_image_url,
                        "spotify_url": spotify_url
                    })
                if song_info:
                    print('songinfo', song_info)
                    return song_info
        
        raise Exception("No suitable playlists found on Spotify.")
    
    except Exception as e:
        print(f"Spotify API error: {e}")
        # Fallback to YouTube with better search terms
        search_term = f"{mood} {language} music"
        videos_search = VideosSearch(search_term, limit=10)
        response = videos_search.result()
        
        if response and "result" in response and response["result"]:
            return [{
                "song": video['title'],
                "image_url": video['thumbnails'][0]['url'] if video['thumbnails'] else "https://via.placeholder.com/300",
                "spotify_url": f"https://www.youtube.com/watch?v={video['id']}"  # Direct YouTube URL
            } for video in response["result"]]
        else:
            # Last resort - return some default songs
            return [{
                "song": "No songs found - Please try a different mood or language",
                "image_url": "https://via.placeholder.com/300",
                "spotify_url": ""
            }]

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
    
    # Add all possible moods for the template
    moods = ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"]
    languages = ["English", "Tamil", "Spanish", "Hindi", "Korean"]
    
    return render_template(
        "recommendations.html", 
        mood=mood, 
        language=language, 
        songs=songs,
        moods=moods,
        languages=languages
    )

if __name__ == "__main__":
    app.run(debug=True)
