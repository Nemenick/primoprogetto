from _Codici2 import get_hos
import numpy as np
import scipy

"""higher order statistics"""
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

