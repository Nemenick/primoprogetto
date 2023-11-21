
import scipy.signal as sc_sig
import numpy as np


def freq_filter(signal,sf,freqs,type_filter="bandpass", order_filter=2):
    """ freqs è la lista di frequenze, 2 per il passabanda  """
    # type_filter: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’

    freqs=np.array(freqs)
    filt_b1,filt_a1=sc_sig.butter(order_filter,freqs/(sf/2),btype=type_filter)
    filtered_sig=sc_sig.filtfilt(filt_b1,filt_a1,sc_sig.detrend(signal))
    return filtered_sig