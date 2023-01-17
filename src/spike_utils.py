import cmath
import numpy as np

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
    Computes the magnitudes of the fourier values of `spikes`
    with respect to time periods `i` for `num_samples` equally
    spaced `i` in [0,1).
    
    Parameters
    ----------
    spikes
        Sorted np.ndarray of timestamps of spikes.
    num_samples
        Integer number of time periods to compute in the interval [0,1]
        
    Returns
    -------
    np.ndarray
        The `num_samples` magnitudes of the fourier values for the evaluated
        time periods.
    """
    magnitudes = []
    time_periods = np.linspace(0, 1, num_samples)
    for T in time_periods:
        magnitudes.append(abs(spike_fourier(spikes, 1/T)))
    return magnitudes

def compute_fourier_freq_profile(spikes, num_samples):
    """
    Computes the magnitudes of the fourier values of `spikes`
    with respect to frequencies `1/i` for `num_samples` equally
    spaced `i` in [0,1).
    
    Parameters
    ----------
    spikes
        Sorted np.ndarray of timestamps of spikes.
    num_samples
        Integer number of frequencies to compute in the interval [0,1]
        
    Returns
    -------
    np.ndarray
        The `num_samples` magnitudes of the fourier values for the evaluated
        frequencies.
    """
    magnitudes = []
    frequencies = 1 / np.linspace(0, 1, num_samples)
    for f in frequencies:
        magnitudes.append(abs(spike_fourier(spikes, f)))
    return magnitudes

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


def compute_isi_wave(isi, spikes, num_samples):
    """
    Computes a cosine wave with the frequency component
    equal to the inverse of the inter-spike interval `isi`.
    It uses the spike train `spikes` to determine the phase.

    Parameters
    ----------
    isi
        Float value representing the inter-spike interval used
        to determine the frequency of the cosine wave to be plotted.
    spikes
        Sorted np.ndarray of timestamps of spikes.
    num_samples
        Number of samples used to create the signal in [0,1]
        
    Returns
    -------
    np.ndarray
        The cosine signal of the fourier component determined by `isi`
    """
    time = np.linspace(0, 1, num_samples)
    freq = 1 / isi
    fourier_value = spike_fourier(spikes, freq)
    phase = cmath.phase(fourier_value)
    wave_signal = np.cos(time * 2 * np.pi * freq + phase)
    return wave_signal