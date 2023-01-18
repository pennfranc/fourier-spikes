import cmath
import numpy as np
from scipy.signal import argrelextrema

def n_uniform_spikes(n_spikes):
    """
    Creates a spike train of `n_spikes` encoded as an array of 
    timestamps. All timestamps are uniformly sampled from [0,1].
    The timestamps are sorted in increasing order.
    
    Parameters
    ----------
    n_spikes
        Integer indicating the number of spikes to be produced.
        
    Returns
    -------
    np.ndarray
        A 1-dimensional numpy array of length `n_spikes` with 
        timestamps between 0 and 1.
    """
    timestamps = np.random.uniform(size=n_spikes)
    return np.sort(timestamps)


def spike_fourier(spikes, freq):
    """
    Computes the fourier space value of the given frequency `freq`
    for a sum of dirac deltas given by the timestamps of `spikes`.
    
    Parameters
    ----------
    spikes
        Sorted np.ndarray of timestamps of spikes.
    freq
        Float indicating the frequency at which the fourier
        value of the spike train should be evaluated.
        
    Returns
    -------
    complex
        The complex value of the given freq in the fourier transform
        of the spikes.
    """

    fourier_sum = 0
    for spike_time in spikes:
        fourier_sum += np.exp(complex(0, -2 * np.pi * freq * spike_time))
    return fourier_sum

def compute_fourier_period_profile(spikes, num_samples):
    """
    Computes the fourier values of `spikes` with respect to time periods
    `i` for `num_samples` equally spaced `i` in [0,1).
    
    Parameters
    ----------
    spikes
        Sorted np.ndarray of timestamps of spikes.
    num_samples
        Integer number of time periods to compute in the interval [0,1]
        
    Returns
    -------
    np.ndarray
        The `num_samples` fourier values for the evaluated frequencies.
    """
    fourier_vals = []
    time_periods = np.linspace(0, 1, num_samples)
    for T in time_periods:
        fourier_vals.append(spike_fourier(spikes, 1/T))
    return np.array(fourier_vals), time_periods

def compute_fourier_freq_profile(spikes, num_samples):
    """
    Computes the fourier values of `spikes` with respect to frequencies
    `1/i` for `num_samples` equally spaced `i` in [0,1).
    
    Parameters
    ----------
    spikes
        Sorted np.ndarray of timestamps of spikes.
    num_samples
        Integer number of frequencies to compute in the interval [0,1]
        
    Returns
    -------
    np.ndarray
        The `num_samples` fourier values for the evaluated frequencies.
    """
    fourier_vals = []
    frequencies = (1 / np.linspace(1 / num_samples, 1, num_samples))[::-1]
    for f in frequencies:
        fourier_vals.append(spike_fourier(spikes, f))
    return np.array(fourier_vals), frequencies

def compute_spike_signal(spikes, num_samples):
    """
    Creates a signal with `num_samples` samples in the
    interval [0,1] which is 0 everywhere except at indices
    that capture a timestamp found in `spikes`, where
    its value is equal to 1.

    Parameters
    ----------
    spikes
        Sorted np.ndarray of timestamps of spikes.
    num_samples
        Number of samples used to create the signal in [0,1]
        
    Returns
    -------
    np.ndarray
        The spike train signal of length `num_samples` indicating
        the presence of a spike at a given timestamp.
    """
    spike_signal = np.zeros(num_samples)
    for spike_time in spikes:
        spike_signal[int(spike_time * num_samples)] = 1
    return spike_signal


def compute_wave(freq, spikes, num_samples):
    """
    Computes a cosine wave with the frequency component
    equal to the inverse of the inter-spike interval `isi`.
    It uses the spike train `spikes` to determine the phase.

    Parameters
    ----------
    freq
        Float value representing the frequency of the cosine
        wave to be plotted.
    spikes
        Sorted np.ndarray of timestamps of spikes.
    num_samples
        Number of samples used to create the signal in [0,1]
        
    Returns
    -------
    np.ndarray
        The cosine signal of the fourier component determined by `freq`
    """
    time = np.linspace(0, 1, num_samples)
    fourier_value = spike_fourier(spikes, freq)
    phase = cmath.phase(fourier_value)
    wave_signal = np.cos(time * 2 * np.pi * freq + phase)
    return wave_signal


def peak_freq_components(spikes, frequency_magnitudes, frequencies, limited_indices=None, eps=0.001):
    """
    Computes peaks in the fourier transform of `spikes` that are within `eps`
    of their maximal value, which corresponds to `len(spikes)`.

    Parameters
    ----------
    spikes
        np.ndarray spike train as sorted time stamps.
    frquency_magnitudes
        np.ndarray of the magnitudes of the fourier values
        of the spikes for each frequency.
    frequencies
        np.ndarray containing the chosen frequencies.
    limit_indices
        Optionally, a list of integers specifying a subset
        of indices of peaks to return

        
    Returns
    -------
    Tuple(np.ndarray, np.ndarray)
        The values and indices of frequencies leading to peak magnitudes
        of the fourier values of `spikes`
    """
    
    # compute threshold for returned peaks
    threshold = len(spikes) - eps

    # set all magnitudes that do not correspond to a local maximum to zero
    non_peak_mask = np.ones(len(frequency_magnitudes))
    non_peak_mask[argrelextrema(frequency_magnitudes, np.greater)[0]] = 0
    frequency_magnitudes_filtered = frequency_magnitudes.copy()
    frequency_magnitudes_filtered[non_peak_mask.astype(bool)] = 0

    # filter frequencies based on epsilon-threshold
    selected_indices = np.where(frequency_magnitudes_filtered > threshold)[0]
    selected_frequencies = frequencies[selected_indices]

    if limited_indices is not None:
        selected_indices = selected_indices[limited_indices]
        selected_frequencies = selected_frequencies[limited_indices]

    return selected_frequencies, selected_indices