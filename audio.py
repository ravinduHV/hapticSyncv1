import sounddevice as sd
import numpy as np

def record_audio(sample_rate=16000, duration=0.5):
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=2, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def preprocess_audio(audio):
    """
    Converts 2-channel stereo audio to mono by averaging the channels.
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("Input must be a 2D array with shape (samples, 2)")
    return np.mean(audio, axis=1)
