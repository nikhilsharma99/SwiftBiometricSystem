import os
import pickle
import numpy as np
from .feature_extraction import FeatureExtractor
import time

def predict_speaker(audio_path, model, scaler, encoder, anti_spoof_model=None):
    """
    Predict speaker from audio file.
    Args:
        audio_path (str): Path to audio file.
        model: Trained speaker verification model.
        scaler: Feature scaler.
        encoder: Label encoder.
        anti_spoof_model: Optional anti-spoofing model.
    Returns:
        str: Predicted speaker label
    """
    start_time = time.time()
    
    # Extract features
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(audio_path)
    
    if features is None:
        return None
        
    # Reshape and scale features
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    # Get speaker name and confidence
    speaker = encoder.inverse_transform(prediction)[0]
    confidence = np.max(probabilities) * 100
    
    # Add timing information
    processing_time = time.time() - start_time
    
    return f"{speaker} (Confidence: {confidence:.2f}%, Time: {processing_time:.2f}s)"

def main():
    # Load the trained model and label encoder
    with open(os.path.join('models', 'voice_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join('models', 'label_encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)
    # TODO: Load the scaler used for feature scaling during training
    scaler = None
    
    # Test directory path
    test_dir = os.path.join('data', 'swift-voice-biometricsdataspeaker-recognition-audio-dataset')
    
    print("Voice Biometric Authentication System")
    print("====================================")
    
    while True:
        print("\nOptions:")
        print("1. Predict from file path")
        print("2. Test with sample from dataset")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            audio_path = input("\nEnter the path to the audio file (.wav): ")
            if not os.path.exists(audio_path):
                print("File not found!")
                continue
                
            result = predict_speaker(audio_path, model, scaler, encoder)
            if result:
                print(f"\nPredicted Speaker: {result}")
            else:
                print("Error processing audio file")
                
        elif choice == '2':
            # Get a random sample from the dataset
            speakers = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            speaker = np.random.choice(speakers)
            speaker_dir = os.path.join(test_dir, speaker)
            audio_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
            test_file = np.random.choice(audio_files)
            test_path = os.path.join(speaker_dir, test_file)
            
            print(f"\nTesting with file: {test_file}")
            print(f"Actual Speaker: {speaker}")
            
            result = predict_speaker(test_path, model, scaler, encoder)
            if result:
                print(f"Predicted Speaker: {result}")
                print(f"Correct: {'✓' if result.split(' ')[0] == speaker else '✗'}")
            else:
                print("Error processing audio file")
                
        elif choice == '3':
            print("\nThank you for using the Voice Biometric System!")
            break
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()

# TODO: Add test cases with real and spoofed audio for anti-spoofing evaluation
