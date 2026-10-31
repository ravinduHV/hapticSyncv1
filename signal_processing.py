from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, lfilter
import numpy as np


def low_pass_filter(data, cutoff=300.0, fs=16000.0, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return lfilter(b, a, data)


def extract_envelope(signal):
    analytic_signal = hilbert(signal)             # Get analytic representation of the signal
    envelope = np.abs(analytic_signal)            # Envelope = magnitude of analytic signal
    filtered = low_pass_filter(envelope, cutoff=300.0)  # Filter out high-frequency noise
    return filtered


def get_frequencies(signal: np.ndarray, fs: float):
    """
    Computes the one-sided frequency spectrum of a signal.
    
    Args:
        signal (np.ndarray): 1D audio signal (left or right channel)
        fs (float): Sampling rate in Hz

    Returns:
        freqs (np.ndarray): Frequencies in Hz
        magnitude (np.ndarray): Magnitude at each frequency
    """
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / fs)
    
    return xf[:n // 2], 2.0 / n * np.abs(yf[:n // 2])
