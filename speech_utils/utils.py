import wave

import numpy as np


def read_file(filename):
    """
    Read wave data from a *.wav file
    """
    with wave.open(filename,'r') as fin:
        params = fin.getparams()
        nchannels, sampwidth, framerate, wav_length = params[:4]
        str_data = fin.readframes(wav_length)
        wavedata = np.fromstring(str_data, dtype=np.short)
        time = np.arange(0, wav_length) * (1.0 / framerate)

    return wavedata, time, framerate, wav_length
