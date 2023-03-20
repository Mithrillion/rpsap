import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def maxNASC(magReadings, win_len=100, lag_range=(40, 100)):
    lagMin, lagMax = lag_range
    maxNacReading = np.zeros(len(magReadings))  # correlation is from 0 to 1
    for lag in range(lagMin, lagMax):
        tempNac = calculateNASC(magReadings, win_len, lag)
        maxNacReading[: len(tempNac)] = np.maximum(
            maxNacReading[: len(tempNac)], tempNac
        )
    return maxNacReading


def calculateNASC(magReadings, win_len, lag):
    sw = sliding_window_view(magReadings, win_len)
    avg = np.mean(sw, -1)
    sd = np.std(sw, -1)

    top = sw[:-lag] - avg[:-lag, None]
    bottom = sw[lag:] - avg[lag:, None]
    dot = np.sum(top * bottom, -1)
    normalization = lag * sd[lag:] * sd[:-lag]
    return np.clip(dot / normalization, 0, 1)
