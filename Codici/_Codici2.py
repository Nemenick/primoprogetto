
from _Codici_sliding_window_view import sliding_window_view
import scipy


def get_hos(data, window_size, func):
    """

    @param data: waveform
    @param window_size: the moving window size of hos function
    @param func: function of hos
    @return: hos
    """

    # detrend the waveform
    data = scipy.signal.detrend(data)

    # get a sliding window view of given np array
    slid_view =  sliding_window_view(data, (window_size,), (1,))

    # apply the function of slid_view along axis 1
    return func(slid_view,axis=1)

