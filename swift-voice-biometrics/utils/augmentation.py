import numpy as np
import librosa
import soundfile as sf

# TODO: Add real noise samples to /data/noise/


def add_background_noise(audio_path, noise_path, snr_db=10, output_path=None):
    """
    Add background noise to an audio file at a given SNR (signal-to-noise ratio).
    Args:
        audio_path (str): Path to clean audio file.
        noise_path (str): Path to noise audio file.
        snr_db (float): Desired SNR in dB.
        output_path (str): Where to save the noisy audio (optional).
    Returns:
        np.ndarray: Noisy audio signal.
    """
    # Load audio and noise
    audio, sr = librosa.load(audio_path, sr=None)
    noise, _ = librosa.load(noise_path, sr=sr)
    # Trim or loop noise to match audio length
    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(audio)]
    # Scale noise to desired SNR
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(audio_power / (snr_linear * noise_power))
    noisy_audio = audio + scale * noise
    # Save if requested
    if output_path:
        sf.write(output_path, noisy_audio, sr)
    return noisy_audio

# Example usage (uncomment for testing):
# noisy = add_background_noise('clean.wav', 'noise.wav', snr_db=10, output_path='noisy.wav')
# TODO: Integrate with data_processing.py for data augmentation 