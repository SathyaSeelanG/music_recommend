import os
from utils.emotion_model import EmotionDetector

def train_and_save_model():
    print("\n=== Starting Model Training ===")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Create dataset structure
    dataset_path = "datasets/train"
    for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
        os.makedirs(os.path.join(dataset_path, emotion), exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Please create dataset directory at {dataset_path}")
        print("Each emotion should have its own subdirectory with training images")
        return
    
    # Initialize and train the model
    emotion_detector = EmotionDetector()
    history = emotion_detector.train(dataset_path)
    
    if history:
        # Save the trained model
        model_path = 'models/emotion_model.h5'
        emotion_detector.model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save training metrics
        with open('models/training_metrics.txt', 'w') as f:
            f.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
    
if __name__ == "__main__":
    train_and_save_model() 