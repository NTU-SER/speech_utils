import wave

import numpy as np


def read_wave_from_file(filepath):
    """Read wave data from a *.wav file.
    Adapted from https://github.com/xuanjihe/speech-emotion-recognition/blob/master/zscore.py

    Parameters
    ----------
    filepath : str
        Path to the *.wav file.

    Returns
    -------
    wave_data: ndarray
        Wave data extracted from the *.wav file.
    ticks: ndarray
        An array representing ticks in the audio sequence.
    frame_rate: int
        Number of frames per second.

    """
    with wave.open(filepath,'r') as fin:
        params = fin.getparams()
        num_channels, samp_width, frame_rate, num_frames = params[:4]
        # Read all available frames
        str_data = fin.readframes(num_frames)
        wave_data = np.fromstring(str_data, dtype=np.short)
        ticks = np.arange(0, num_frames) * (1.0 / frame_rate)
    return wave_data, ticks, frame_rate
