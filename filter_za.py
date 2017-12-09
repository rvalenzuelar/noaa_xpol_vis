"""
    Filter X-Pol ZA field

    source: https://fiiir.com

"""

def filterx(s):

    import numpy as np

    # Example code, computes the coefficients of a high-pass
    # windowed-sinc filter.

    # Configuration.
    fS = 5  # Sampling rate.
    fH = 1.25  # Cutoff frequency.
    N = 31  # Filter length, must be odd.

    # Compute sinc filter.
    h = np.sinc(2 * fH / fS * (np.arange(N) - (N - 1) / 2.))

    # Apply window.
    h *= np.hamming(N)

    # Normalize to get unity gain.
    h /= np.sum(h)

    # Create a high-pass filter from the low-pass filter
    # through spectral inversion.
    h = -h
    h[(N - 1) / 2] += 1

    print(h)

    # Applying the filter to a signal s can be as simple as writing
    s = np.convolve(s, h)

    return s