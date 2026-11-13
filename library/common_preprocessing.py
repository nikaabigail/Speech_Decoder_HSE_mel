"""Retained historical record-wise ECoG and MEL40 preprocessing."""
import numpy as np
import librosa
import sklearn.preprocessing
import scipy.signal


def notch_filtering_advanced(ecog, frequency):
    ecog_filtered = ecog
    w0_narrow = set(range(100, 200, 50)).union(set(range(30, 300, 30)), [70])
    w0_wide = set(range(200, 350, 50)).union(set([50]))
    for w0 in w0_narrow:
        norch_b, norch_a = scipy.signal.iirnotch(w0, Q=w0, fs=frequency)
        ecog_filtered = scipy.signal.filtfilt(norch_b, norch_a, ecog_filtered, axis=0)
    for w0 in w0_wide:
        #print(w0)
        norch_b, norch_a = scipy.signal.iirnotch(w0, Q=w0//2, fs=frequency)
        ecog_filtered = scipy.signal.filtfilt(norch_b, norch_a, ecog_filtered, axis=0)
    return ecog_filtered


def remove_eyes_artifacts(ecog, frequency, high_pass_frequency):
    bgamma, agamma = scipy.signal.butter(5, high_pass_frequency / (frequency / 2), btype='high')
    return scipy.signal.filtfilt(bgamma, agamma, ecog, axis=0)


def remove_target_leakage(ecog, frequency, low_pass_frequency):
    bgamma, agamma = scipy.signal.butter(5, low_pass_frequency / (frequency / 2), btype='low')
    return scipy.signal.filtfilt(bgamma, agamma, ecog, axis=0)


def extract_melspectrogram(sound, sampling_rate, downsampling_coef, out_length, f_max, n_mels):
    melspectrogram = librosa.feature.melspectrogram(
        y=sound, sr=sampling_rate, n_mels=n_mels, fmax=f_max, hop_length=downsampling_coef,
    )
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max).T
    # todo: remove this terrible ifs
    if melspectrogram.shape[0] == out_length:
        return melspectrogram
    elif melspectrogram.shape[0] - 1 == out_length:
        return melspectrogram[:-1]
    else:
        raise ValueError
    return


def classic_ecog_pipeline(ecog, sampling_rate, downsampling_coef, low_pass_hz, high_pass_hz):
    ecog = scipy.signal.decimate(ecog, downsampling_coef, axis=0)
    ecog = remove_eyes_artifacts(ecog, int(sampling_rate / downsampling_coef), high_pass_hz)
    ecog = notch_filtering_advanced(ecog, int(sampling_rate / downsampling_coef))
    ecog = remove_target_leakage(ecog, int(sampling_rate / downsampling_coef), low_pass_hz)
    ecog = sklearn.preprocessing.scale(ecog, copy=False)
    return ecog


def classic_melspectrogram_pipeline(sound, sampling_rate, downsampling_coef, ecog_size, n_mels, f_max):
    sound /= np.max(np.abs(sound))
    melspectrogram = extract_melspectrogram(sound, sampling_rate, downsampling_coef, ecog_size, f_max, n_mels)
    melspectrogram = sklearn.preprocessing.scale(melspectrogram)
    return melspectrogram
