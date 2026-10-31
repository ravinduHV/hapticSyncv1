from model import RecognitionModel
from audio import preprocess_audio, record_audio
from signal_processing import get_frequencies

# Change the variables accordingly
SAMPLE_RATE = 16000
DURATION = 0.125

model = RecognitionModel()

while True:
    audio = record_audio()

    freqs_left, mags_left = get_frequencies(audio[:, 0], fs=16000)
    freqs_right, mags_right = get_frequencies(audio[:, 1], fs=16000)
    
    phonemes = model.predict(preprocess_audio(audio))
    print("🔊 Frequencies (Left Channel):", freqs_left)
    print("🔊 Magnitudes (Left Channel):", mags_left)
    print("🔊 Frequencies (Right Channel):", freqs_right)
    print("🔊 Magnitudes (Right Channel):", mags_right)
    print("📣 Recognized phonemes:", phonemes)
