import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_recommendations():
    """Analyze recommendation patterns and user preferences."""
    user_prefs = pd.read_csv('data/user_preferences.csv')
    
    # Analyze emotion-music correlations
    emotion_correlations = user_prefs.groupby('emotion')[
        ['valence', 'energy', 'danceability']
    ].mean()
    
    # Plot insights
    plt.figure(figsize=(10, 6))
    sns.heatmap(emotion_correlations, annot=True)
    plt.title('Emotion-Music Feature Correlations')
    plt.savefig('static/analysis/emotion_correlations.png') 