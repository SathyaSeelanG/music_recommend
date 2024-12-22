import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SongFeatureExtractor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def extract_features(self, song):
        """Extract relevant features from a song."""
        features = np.array([
            song.get('valence', 0),
            song.get('energy', 0),
            song.get('danceability', 0),
            song.get('tempo', 0) / 200,  # Normalize tempo
            song.get('mode', 0),
            song.get('liveness', 0),
            song.get('acousticness', 0)
        ])
        return self.scaler.fit_transform(features.reshape(1, -1))[0]

    def get_state(self, song_history):
        """Create state from last 5 songs."""
        state = []
        for song in song_history[-5:]:
            state.extend(self.extract_features(song))
        # Pad if less than 5 songs
        while len(state) < 35:  # 7 features * 5 songs
            state.extend([0] * 7)
        return np.array(state).reshape(1, -1) 