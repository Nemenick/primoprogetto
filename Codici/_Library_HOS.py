import scipy.signal as sc_sig
import numpy as np
import scipy

def freq_filter(signal,sf,freqs,type_filter="bandpass", order_filter=2):
    """ freqs: list of frequences (e.g. 2 for bandpass), or single float (e.g. for highpass)  """
    # type_filter: ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’

    freqs=np.array(freqs)
    filt_b1,filt_a1=sc_sig.butter(order_filter,freqs/(sf/2),btype=type_filter)
    filtered_sig=sc_sig.filtfilt(filt_b1,filt_a1,sc_sig.detrend(signal))
    return filtered_sig

def sliding_window_view(arr, window_shape, steps):
    # -*- coding: utf-8 -*-
    """
    Created on 4/6/2021
    @author: Ryan meowklaski
    """

    """ Produce a view from a sliding, striding window over `arr`.
        The window is only placed in 'valid' positions - no overlapping
        over the boundary.
        Parameters
        ----------
        arr : numpy.ndarray, shape=(...,[x, (...), z])
            The array to slide the window over.
        window_shape : Sequence[int]
            The shape of the window to raster: [Wx, (...), Wz],
            determines the length of [x, (...), z]
        steps : Sequence[int]
            The step size used when applying the window
            along the [x, (...), z] directions: [Sx, (...), Sz]
        Returns
        -------
        view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
            Where X = (x - Wx) // Sx + 1
        Notes
        -----
        In general, given
          `out` = sliding_window_view(arr,
                                      window_shape=[Wx, (...), Wz],
                                      steps=[Sx, (...), Sz])
           out[ix, (...), iz] = arr[..., ix*Sx:ix*Sx+Wx,  (...), iz*Sz:iz*Sz+Wz]
         Examples
         --------
         >>> import numpy as np
         >>> x = np.arange(9).reshape(3,3)
         >>> x
         array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
         >>> y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
         >>> y
         array([[[[0, 1],
                  [3, 4]],
                 [[1, 2],
                  [4, 5]]],
                [[[3, 4],
                  [6, 7]],
                 [[4, 5],
                  [7, 8]]]])
        >>> np.shares_memory(x, y)
         True
        # Performing a neural net style 2D conv (correlation)
        # placing a 4x4 filter with stride-1
        >>> data = np.random.rand(10, 3, 16, 16)  # (N, C, H, W)
        >>> filters = np.random.rand(5, 3, 4, 4)  # (F, C, Hf, Wf)
        >>> windowed_data = sliding_window_view(data,
        ...                                     window_shape=(4, 4),
        ...                                     steps=(1, 1))
        >>> conv_out = np.tensordot(filters,
        ...                         windowed_data,
        ...                         axes=[[1,2,3], [3,4,5]])
        # (F, H', W', N) -> (N, F, H', W')
        >>> conv_out = conv_out.transpose([3,0,1,2])
         """
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
    in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
    window_shape = np.array(window_shape)  # [Wx, (...), Wz]
    steps = np.array(steps)  # [Sx, (...), Sz]
    nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

    # number of per-byte steps to take to fill window
    window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
    # number of per-byte steps to take to place window
    step_strides = tuple(window_strides[-len(steps):] * steps)
    # number of bytes to step to populate sliding window view
    strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

    outshape = tuple((in_shape - window_shape) // steps + 1)
    # outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
    outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
    return as_strided(arr, shape=outshape, strides=strides, writeable=False)

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



"""Higher Order Statistics"""
def S_1(data, **kwargs):
    return np.mean(data,**kwargs)

def S_2(data,**kwargs):
    return np.std(data,ddof=1,**kwargs)

def S_3(data,**kwargs):
    return scipy.stats.skew(data,**kwargs)

def S_4(data,axis=1,**kwargs):
    return scipy.stats.kurtosis(data,axis=axis,**kwargs)

def S_6(data, axis=1):
    # can be very slow
    return np.sum((data-np.mean(data,axis=axis)[:,None])**6,axis=axis)/(data.shape[1]-1)/np.std(data,ddof=1,axis=axis)**6-15

def get_onset(waveform,window_size=100, threshold=0.1, statistics=S_6):
    # use hos to pick onset

    # get hos, here we use S 4
    hos = get_hos(waveform, window_size, statistics)
    # smooth the S4
    hos = np.convolve(hos, np.ones(3)/3, mode='valid')
    # get first derivative
    diff = np.diff(hos)
    # narrow the search range to a region near the maximum
    pre_window = 100
    # lower_bound = np.argmax(np.abs(waveform)) - 10 - window_size
    # upper_bound = len(waveform) - 10
    lower_bound = 1
    upper_bound = len(waveform)

    #lower_bound = np.argmax(hos[lower_bound:upper_bound]) + lower_bound - pre_window
    try:
        # find the onset larger than 0.1 * maximum of diff
        onset = np.where(diff[lower_bound:upper_bound] > threshold * np.max(diff))[0][0] + lower_bound
        onset_2 = np.argmax(diff[lower_bound:upper_bound])

    except:
        #try:
            # try 2 times the threshold in later part
            # onset = np.where(diff[lower_bound:] > 2*threshold * np.max(diff))[0][0] + lower_bound
        onset = -1 - window_size
        onset_2 = onset
        # except:
        #     # use trigger position when nothing found
        #     onset = 1280

    return onset + window_size//2, diff, onset_2 + window_size//2



def get_onset_2(waveform,window_size=100, threshold=0.1, statistics=S_6):
    # Cannot use the get_onset without upper and lower bounds, original search window for DETECT waveforms is too large

    # get hos, here we use S 4
    hos = get_hos(waveform, window_size, statistics)
    # smooth the S4
    hos = np.convolve(hos, np.ones(3)/3, mode='valid')
    # get first derivative
    diff = np.diff(hos)
    # narrow the search range to a region near the maximum
    pre_window = 100
    lower_bound = np.argmax(np.abs(waveform)) - 200 - window_size
    upper_bound = lower_bound + 400
    lower_bound = np.argmax(hos[lower_bound:upper_bound]) + lower_bound - pre_window
    try:
        # find the onset larger than 0.1 * maximum of diff
        onset = np.where(diff[lower_bound:upper_bound] > threshold * np.max(diff))[0][0] + lower_bound
    except:
        try:
            # try 2 times the threshold in later part
            onset = np.where(diff[lower_bound:] > 2*threshold * np.max(diff))[0][0] + lower_bound

        except:
            # use trigger position when nothing found
            onset = 1280

    return onset + window_size//2, lower_bound