import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class MusicDataProcessor:
    def __init__(self):
        self.emotion_mapping = pd.read_csv('data/emotion_music_mapping.csv')
        self.tracks_data = pd.read_csv('data/spotify_tracks.csv')
        self.scaler = MinMaxScaler()
        
    def get_emotion_features(self, emotion):
        """Get music features for detected emotion."""
        features = self.emotion_mapping[self.emotion_mapping['emotion'] == emotion].iloc[0]
        return features[['valence', 'energy', 'danceability', 'tempo', 'mode']]
    
    def find_matching_songs(self, emotion, language, n_songs=10):
        """Find songs matching emotional features."""
        emotion_features = self.get_emotion_features(emotion)
        
        # Scale features
        track_features = self.tracks_data[['valence', 'energy', 'danceability', 'tempo', 'mode']]
        scaled_features = self.scaler.fit_transform(track_features)
        emotion_features_scaled = self.scaler.transform([emotion_features])
        
        # Calculate similarity
        similarities = cosine_similarity(emotion_features_scaled, scaled_features)[0]
        
        # Get top matching tracks
        top_indices = np.argsort(similarities)[-n_songs:][::-1]
        recommended_tracks = self.tracks_data.iloc[top_indices]
        
        return recommended_tracks 