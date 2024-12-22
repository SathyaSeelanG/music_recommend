import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

class EmotionDetector:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.model = self._load_or_build_model()
        
    def _load_or_build_model(self):
        model_path = 'models/emotion_model.h5'
        if os.path.exists(model_path):
            print("Loading pre-trained model...")
            return tf.keras.models.load_model(model_path)
        else:
            print("Building new model...")
            return self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def predict_emotion(self, image):
        try:
            # Enhanced preprocessing
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = clahe.apply(img)
            
            # Normalize the image
            img = cv2.resize(img, (48, 48))
            img = img.astype('float32') / 255.0
            
            # Add Gaussian blur to reduce noise
            img = cv2.GaussianBlur(img, (3,3), 0)
            
            img = img.reshape(1, 48, 48, 1)
            
            # Get prediction with confidence scores
            prediction = self.model.predict(img, verbose=0)
            
            # Get confidence scores for each emotion
            emotion_scores = [(emotion, conf*100) for emotion, conf in zip(self.emotions, prediction[0])]
            # Sort by confidence in descending order
            emotion_scores.sort(key=lambda x: x[1], reverse=True)
            
            print("\n=== Emotion Detection Results ===")
            print("Confidence scores for each emotion:")
            
            # Print all scores
            for emotion, conf in emotion_scores:
                print(f"{emotion}: {conf:.2f}%")
            
            # Get top 2 emotions
            top_emotion, top_conf = emotion_scores[0]
            second_emotion, second_conf = emotion_scores[1]
            
            # Define minimum confidence threshold
            MIN_CONFIDENCE = 20.0  # 20%
            
            # If all confidences are too close (within 1% of each other)
            confidence_range = emotion_scores[0][1] - emotion_scores[-1][1]
            if confidence_range < 1.0:
                print("\nConfidence scores too close together - defaulting to neutral")
                return "neutral", emotion_scores[0][1]/100
            
            # If top confidence is below minimum threshold
            if top_conf < MIN_CONFIDENCE:
                print(f"\nLow confidence detection ({top_conf:.2f}% < {MIN_CONFIDENCE}%) - defaulting to neutral")
                return "neutral", top_conf/100
            
            # If second emotion is close to top emotion (within 5%)
            if (top_conf - second_conf) < 5.0:
                # Prefer more common emotions when confidences are close
                preferred_emotions = ['happy', 'sad', 'neutral']
                for emotion in preferred_emotions:
                    if top_emotion == emotion or second_emotion == emotion:
                        selected_emotion = emotion
                        selected_conf = next(conf for em, conf in emotion_scores if em == emotion)
                        print(f"\nClose confidences - selecting preferred emotion: {selected_emotion}")
                        return selected_emotion, selected_conf/100
            
            print(f"\nSelected Emotion: {top_emotion} ({top_conf:.2f}% confidence)")
            return top_emotion, top_conf/100
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "neutral", 0.0
    
    def train(self, train_dir, validation_split=0.2, epochs=20):
        """Train the emotion detection model."""
        try:
            X = []
            y = []
            
            print("\nLoading dataset...")
            
            # Count images per emotion
            emotion_counts = {emotion: 0 for emotion in self.emotions}
            
            # Load and preprocess training data
            for emotion_idx, emotion in enumerate(self.emotions):
                emotion_dir = os.path.join(train_dir, emotion)
                if os.path.exists(emotion_dir):
                    files = os.listdir(emotion_dir)
                    emotion_counts[emotion] = len(files)
                    print(f"Found {len(files)} images for {emotion}")
                    
                    for image_file in files:
                        img_path = os.path.join(emotion_dir, image_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (48, 48))
                            X.append(img)
                            y.append(emotion_idx)
            
            if not X:
                raise ValueError("No training images found!")
            
            print("\nDataset Statistics:")
            print(f"Total images: {len(X)}")
            for emotion, count in emotion_counts.items():
                print(f"{emotion}: {count} images ({count/len(X)*100:.1f}%)")
            
            # Convert to numpy arrays
            X = np.array(X).astype('float32') / 255.0
            X = X.reshape(X.shape[0], 48, 48, 1)
            y = tf.keras.utils.to_categorical(y, len(self.emotions))
            
            # Train the model
            print("\nTraining model...")
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            return history
            
        except Exception as e:
            print(f"Error during training: {e}")
            return None