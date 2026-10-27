"""
ppg.py - core signal-processing functions for heart-rate estimation from a PPG signal.

    bandpass()      zero-phase Butterworth band-pass filter
    detect_peaks()  systolic peak detection
    peaks_hr()      heart rate from inter-beat intervals (time domain)
    fft_hr()        heart rate from the dominant spectral peak (frequency domain)
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass(x, fs, low=0.5, high=4.0, order=3):
    """Keep only 0.5-4 Hz (30-240 bpm). filtfilt runs the filter forwards and
    backwards, so the peaks are not shifted in time."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def detect_peaks(x_filt, fs, max_bpm=200, prominence_factor=0.3):
    """Find heartbeats. Two beats cannot be closer than 60/max_bpm seconds,
    and a peak must stand out from its surroundings by some fraction of the signal SD."""
    min_distance = max(1, int(fs * 60.0 / max_bpm))
    prominence = prominence_factor * np.std(x_filt)
    peaks, _ = find_peaks(x_filt, distance=min_distance, prominence=prominence)
    return peaks


def peaks_hr(t, peaks, min_bpm=40, max_bpm=200):
    """Beat-by-beat heart rate from inter-beat intervals; drop implausible values.
    Returns (times, hr_values). Uses the median to be robust to a missed/extra beat."""
    if len(peaks) < 2:
        return np.array([]), np.array([])
    ibi = np.diff(t[peaks])
    hr = 60.0 / ibi
    ok = (hr >= min_bpm) & (hr <= max_bpm)
    return t[peaks][1:][ok], hr[ok]


def fft_hr(x_filt, fs, low=0.5, high=4.0):
    """Dominant frequency in the heart-rate band, converted to bpm."""
    n = len(x_filt)
    spec = np.abs(np.fft.rfft(x_filt * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        return np.nan
    return 60.0 * freqs[band][np.argmax(spec[band])]
