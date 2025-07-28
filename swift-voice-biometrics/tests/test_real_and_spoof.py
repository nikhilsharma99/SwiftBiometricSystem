import sys
import os
import joblib
import argparse
from datetime import datetime
import time

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.predict import predict_speaker

# Configuration
CONFIDENCE_THRESHOLD = 70  # Minimum confidence level for reliable prediction

def load_models():
    """Load all required models"""
    base_path = os.path.join(os.path.dirname(__file__), '..')
    try:
        model = joblib.load(os.path.join(base_path, 'models/voice_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'models/scaler.pkl'))
        encoder = joblib.load(os.path.join(base_path, 'models/label_encoder.pkl'))
        return model, scaler, encoder
    except Exception as e:
        print(f"[ERROR] Failed to load models: {str(e)}")
        return None, None, None

def test_single_file(audio_path):
    """Test a single audio file for speaker recognition"""
    if not os.path.exists(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        return None
    
    try:
        # Load models
        model, scaler, encoder = load_models()
        if model is None:
            return None

        # Get prediction
        result = predict_speaker(audio_path, model, scaler, encoder)
        
        # Parse the result string to get confidence
        confidence = float(result.split("Confidence: ")[1].split("%")[0])
        
        # Determine reliability based on confidence
        reliability = "HIGH" if confidence >= CONFIDENCE_THRESHOLD else "LOW"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log the result with more details
        log_path = os.path.join(os.path.dirname(__file__), '../docs/real_world_test_log.md')
        with open(log_path, 'a') as f:
            f.write(f"\n| {os.path.basename(audio_path)} | {result} | Reliability: {reliability} | {timestamp} |")
        
        return result, reliability
    except Exception as e:
        print(f"[ERROR] Failed to process {audio_path}: {str(e)}")
        return None

def test_batch_files(audio_files):
    """Test multiple audio files and compare results"""
    results = []
    print("\nBatch Testing Results:")
    print("-" * 80)
    
    for audio_path in audio_files:
        result = test_single_file(audio_path)
        if result:
            result_str, reliability = result
            results.append({
                'file': os.path.basename(audio_path),
                'result': result_str,
                'reliability': reliability
            })
            print(f"\nFile: {os.path.basename(audio_path)}")
            print(f"Result: {result_str}")
            print(f"Reliability: {reliability}")
            print("-" * 40)
    
    return results

def run_comparison_test():
    """Run comparison tests with different emotional states"""
    base_path = os.path.join(os.path.dirname(__file__), '..')
    test_files = [
        os.path.join(base_path, 'data/swift-voice-biometricsdataspeaker-recognition-audio-dataset/OAF_neutral/OAF_back_neutral.wav'),
        os.path.join(base_path, 'data/swift-voice-biometricsdataspeaker-recognition-audio-dataset/OAF_happy/OAF_back_happy.wav'),
        os.path.join(base_path, 'data/swift-voice-biometricsdataspeaker-recognition-audio-dataset/OAF_angry/OAF_back_angry.wav'),
        os.path.join(base_path, 'data/swift-voice-biometricsdataspeaker-recognition-audio-dataset/OAF_Sad/OAF_back_sad.wav')
    ]
    
    print("\nRunning Comparison Test with Different Emotional States...")
    print("Testing the same word 'back' with neutral, happy, angry, and sad emotions")
    results = test_batch_files(test_files)
    
    # Analyze consistency
    if results:
        speakers = [r['result'].split()[0] for r in results]  # Get predicted speaker names
        is_consistent = all(s == speakers[0] for s in speakers)
        
        print("\nConsistency Analysis:")
        print("-" * 80)
        print(f"Speaker Recognition Consistency: {'HIGH' if is_consistent else 'LOW'}")
        if not is_consistent:
            print("\n[WARNING] Inconsistent speaker recognition across emotional states!")
            print("This might indicate that emotional variations are affecting recognition accuracy.")
            print_test_guidelines()
    
    return results

def initialize_log():
    """Initialize or update the test log file"""
    log_path = os.path.join(os.path.dirname(__file__), '../docs/real_world_test_log.md')
    
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("# Voice Recognition Test Results\n\n")
            f.write("| Audio File | Predicted Speaker | Reliability | Timestamp |\n")
            f.write("|------------|------------------|-------------|------------|\n")

def print_test_guidelines():
    """Print guidelines for getting better results"""
    print("\nGuidelines for best results:")
    print("1. Use high-quality audio recordings (16kHz+ sample rate)")
    print("2. Ensure clear speech with minimal background noise")
    print("3. Record in a quiet environment")
    print("4. Keep consistent distance from microphone")
    print("5. Speak naturally at a normal pace")
    print("\nReliability Levels:")
    print(f"- HIGH: Confidence >= {CONFIDENCE_THRESHOLD}%")
    print(f"- LOW: Confidence < {CONFIDENCE_THRESHOLD}%")

def run_guided_test():
    """Run a guided test with step-by-step instructions"""
    print("\n=== Voice Biometrics Testing Guide ===")
    print("\nThis guide will help you test voice recognition step by step.")
    
    # Step 1: Environment Check
    print("\nStep 1: Environment Setup")
    print("-------------------------")
    print("1. Find a quiet environment")
    print("2. Minimize background noise")
    print("3. Prepare your microphone")
    input("Press Enter when you're ready...")

    # Step 2: Sample Tests
    print("\nStep 2: Testing with Sample Files")
    print("--------------------------------")
    print("We'll test the system with sample files from our dataset.")
    
    base_path = os.path.join(os.path.dirname(__file__), '..')
    sample_tests = [
        {
            'name': 'Neutral Speech',
            'path': os.path.join(base_path, 'data/swift-voice-biometricsdataspeaker-recognition-audio-dataset/OAF_neutral/OAF_back_neutral.wav'),
            'description': 'A neutral voice saying "back"'
        },
        {
            'name': 'Happy Speech',
            'path': os.path.join(base_path, 'data/swift-voice-biometricsdataspeaker-recognition-audio-dataset/OAF_happy/OAF_back_happy.wav'),
            'description': 'A happy voice saying "back"'
        }
    ]

    for test in sample_tests:
        print(f"\nTesting {test['name']}...")
        print(f"Description: {test['description']}")
        result = test_single_file(test['path'])
        if result:
            result_str, reliability = result
            print(f"Result: {result_str}")
            print(f"Reliability: {reliability}")
        time.sleep(1)  # Pause between tests

    # Step 3: Emotional Variation Test
    print("\nStep 3: Testing Emotional Variations")
    print("---------------------------------")
    print("Now we'll test how different emotions affect recognition.")
    input("Press Enter to start emotional variation test...")
    
    run_comparison_test()

    # Step 4: Results Analysis
    print("\nStep 4: Understanding the Results")
    print("-------------------------------")
    print("Key points to understand:")
    print("1. Confidence Score: Higher is better (>70% is considered reliable)")
    print("2. Emotional Impact: Voice should be recognized across different emotions")
    print("3. Processing Time: Initial test may be slower due to model loading")
    
    # Step 5: Recommendations
    print("\nStep 5: Recommendations")
    print("---------------------")
    print_test_guidelines()
    
    # Final Notes
    print("\nTesting Complete!")
    print("You can find detailed logs in: docs/real_world_test_log.md")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test voice recognition on WAV files')
    parser.add_argument('wav_file', nargs='?', help='Path to the WAV file to test')
    parser.add_argument('--comparison', action='store_true', help='Run comparison test with different emotional states')
    parser.add_argument('--guide', action='store_true', help='Run guided test with step-by-step instructions')
    args = parser.parse_args()

    initialize_log()

    if args.guide:
        run_guided_test()
    elif args.comparison:
        run_comparison_test()
    elif args.wav_file:
        # Test single file mode
        result = test_single_file(args.wav_file)
        if result:
            result_str, reliability = result
            print(f"\nResults for {os.path.basename(args.wav_file)}:")
            print(f"Predicted Speaker: {result_str}")
            print(f"Reliability: {reliability}")
            
            # Show guidelines if reliability is LOW
            if reliability == "LOW":
                print("\n[WARNING] Low confidence detection!")
                print_test_guidelines()
    else:
        print("Please choose a testing mode:")
        print("1. Guided Test (recommended for first-time users):")
        print("   python test_real_and_spoof.py --guide")
        print("\n2. Single File Test:")
        print("   python test_real_and_spoof.py <path_to_wav_file>")
        print("\n3. Emotion Comparison Test:")
        print("   python test_real_and_spoof.py --comparison")
        print_test_guidelines()