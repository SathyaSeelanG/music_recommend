import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from youtubesearchpython import VideosSearch
from utils.data_processor import MusicDataProcessor
import pandas as pd
from utils.rl_agent import MusicRLAgent
from utils.feature_extractor import SongFeatureExtractor
from utils.emotion_model import EmotionDetector
# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Initialize Spotify client
spotify = Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# Initialize data processor
music_processor = MusicDataProcessor()

# Initialize RL components
feature_extractor = SongFeatureExtractor()
rl_agent = MusicRLAgent(state_size=35, action_size=len(music_processor.tracks_data))

# Initialize the emotion detector (it will load the saved model)
emotion_detector = EmotionDetector()

def capture_mood():
    """Capture face and detect mood using custom CNN model."""
    cap = cv2.VideoCapture(0)
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Add eye cascade for better face alignment
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    print("\n=== Starting Mood Detection ===")
    print("Opening camera...")
    
    # Set window properties
    cv2.namedWindow("Capturing your mood...", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Capturing your mood...", 0, 0)
    cv2.setWindowProperty("Capturing your mood...", cv2.WND_PROP_TOPMOST, 1)
    
    mood_detected = "neutral"  # Default mood
    countdown = 3
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access camera")
            break
        
        # Calculate remaining time
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        remaining_time = max(0, countdown - int(elapsed_time))
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            print(f"Face detected! ({len(faces)} faces)")
        
        # Draw rectangle around face and add countdown text
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        if remaining_time > 0:
            text = f"Capturing in {remaining_time}..."
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Capturing your mood...", frame)
        
        # Capture and predict when countdown ends
        if elapsed_time >= countdown:
            if len(faces) > 0:
                print("\nAnalyzing facial expression...")
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                
                # Detect eyes for alignment
                eyes = eye_cascade.detectMultiScale(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))
                if len(eyes) >= 2:
                    # Sort eyes by x-coordinate
                    eyes = sorted(eyes, key=lambda x: x[0])
                    
                    # Calculate angle for alignment
                    eye1_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
                    eye2_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
                    angle = np.degrees(np.arctan2(eye2_center[1] - eye1_center[1], 
                                                eye2_center[0] - eye1_center[0]))
                    
                    # Rotate image to align eyes horizontally
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                    face_img = cv2.warpAffine(face_img, M, (w, h))
                
                mood, confidence = emotion_detector.predict_emotion(face_img)
                mood_detected = mood
                print(f"Detected emotion: {mood} ({confidence*100:.2f}%)")
            else:
                print("No face detected in final frame")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detection cancelled by user")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    for i in range(4):
        cv2.waitKey(1)
    
    print(f"\nFinal detected mood: {mood_detected}")
    return mood_detected

def get_music_recommendations(mood, language="English"):
    """Enhanced recommendation function using both Spotify and YouTube."""
    songs = []
    
    try:
        # First try Spotify
        search_query = f"{mood} {language} music"
        print(f"\n=== Searching Spotify ===")
        print(f"Query: {search_query}")
        
        results = spotify.search(q=search_query, type='track', limit=15)
        
        if results['tracks']['items']:
            print(f"\nFound {len(results['tracks']['items'])} Spotify tracks")
            for track in results['tracks']['items']:
                song_info = {
                    "song": f"{track['name']} - {track['artists'][0]['name']}",
                    "image_url": track['album']['images'][0]['url'] if track['album']['images'] else "https://via.placeholder.com/300",
                    "spotify_url": track['external_urls']['spotify'],  # Keep Spotify URL
                    "source": "Spotify",
                    "features": {
                        "valence": 0.5,
                        "energy": 0.5,
                        "danceability": 0.5
                    }
                }
                print(f"Spotify Track: {song_info['song']}")
                songs.append(song_info)
    except Exception as e:
        print(f"Spotify error: {e}")

    try:
        # Also get YouTube results
        print(f"\n=== Searching YouTube ===")
        search_term = f"{mood} {language} music"
        print(f"Query: {search_term}")
        
        videos_search = VideosSearch(search_term, limit=15)
        response = videos_search.result()
        
        if response and "result" in response:
            print(f"\nFound {len(response['result'])} YouTube videos")
            for video in response["result"]:
                song_info = {
                    "song": video['title'],
                    "image_url": video['thumbnails'][0]['url'] if video['thumbnails'] else "https://via.placeholder.com/300",
                    "spotify_url": f"https://www.youtube.com/watch?v={video['id']}",
                    "source": "YouTube",
                    "features": {
                        "valence": 0.5,
                        "energy": 0.5,
                        "danceability": 0.5
                    }
                }
                print(f"YouTube Track: {song_info['song']}")
                songs.append(song_info)
    except Exception as e:
        print(f"YouTube error: {e}")

    if not songs:
        print("\nNo songs found from either source")
        return [{
            "song": "No songs found - Please try different options",
            "image_url": "https://via.placeholder.com/300",
            "spotify_url": "",
            "source": "None",
            "features": {
                "valence": 0.5,
                "energy": 0.5,
                "danceability": 0.5
            }
        }]

    print(f"\nTotal songs found: {len(songs)}")
    return songs

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

@app.route("/feedback", methods=["POST"])
def feedback():
    """Handle user feedback for RL training."""
    data = request.json
    song_id = data.get('song_id')
    liked = data.get('liked', False)
    
    # Get song history
    song_history = session.get('song_history', [])
    if len(song_history) >= 2:
        current_state = feature_extractor.get_state(song_history[:-1])
        next_state = feature_extractor.get_state(song_history)
        
        # Calculate reward based on user feedback
        reward = 1 if liked else -1
        
        # Store experience in agent's memory
        rl_agent.remember(
            current_state,
            song_id,
            reward,
            next_state,
            False
        )
        
        # Train the agent
        if len(rl_agent.memory) > 32:
            rl_agent.replay(32)
    
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
