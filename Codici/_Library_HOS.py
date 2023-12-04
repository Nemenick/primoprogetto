import scipy.signal as sc_sig
import numpy as np
import scipy
from scipy.cluster.hierarchy import linkage, fcluster

def freq_filter(signal,sf,freqs,type_filter="bandpass", order_filter=2):
    """ freqs: list of frequences (e.g. 2 for bandpass), or single float (e.g. for highpass)
        sf sampling frequence  """
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
        onset = -1 - window_size//2
        onset_2 = onset
        # except:
        #     # use trigger position when nothing found
        #     onset = 1280

    return onset + window_size//2, diff, onset_2 + window_size//2

def get_onset_2(waveform,window_size=100, threshold=0.1, statistics=S_6):
    # Cannot use the get_onset without upper and lower bounds, original search window for DETECT waveforms is too large?

    # get hos, here we use S 4
    hos = get_hos(waveform, window_size, statistics)
    # smooth the S4
    hos = np.convolve(hos, np.ones(3)/3, mode='valid')
    # get first derivative
    diff = np.diff(hos)
    # narrow the search range to a region near the maximum
    pre_window = 100
    # lower_bound = np.argmax(np.abs(waveform)) - 200 - window_size
    # upper_bound = lower_bound + 400
    # lower_bound = np.argmax(hos[lower_bound:upper_bound]) + lower_bound - pre_window
    lower_bound = 1
    upper_bound = len(waveform)
    try:
        # find the onset larger than 0.1 * maximum of diff
        onset = np.where(diff[lower_bound:upper_bound] > threshold * np.max(diff))[0][0] + lower_bound
    except:
        try:
            # try 2 times the threshold in later part
            onset = np.where(diff[lower_bound:] > 2*threshold * np.max(diff))[0][0] + lower_bound

        except:
            # use trigger position when nothing found
            onset = -1 - window_size
    onset_2 = np.argmax(diff[lower_bound:upper_bound])

    return onset + window_size, diff, onset_2 + window_size, lower_bound

def get_onset_ori_2(waveform,window_size=100, threshold=0.1, statistics=S_6):
    # good for Pollino waveforms

    # get hos, here we use S 4
    hos = get_hos(waveform, window_size, statistics)
    # smooth the S4
    hos = np.convolve(hos, np.ones(3)/3, mode='valid')
    # get first derivative
    diff = np.diff(hos)
    # narrow the search range to a region near the maximum
    pre_window = 1000
    #lower_bound = np.argmax(np.abs(waveform)) - 200 - window_size
    upper_bound = np.argmax(np.abs(waveform)) + pre_window
    lower_bound = np.argmax(np.abs(waveform)) - pre_window
    try:
        # find the onset larger than 0.1 * maximum of diff
        onset = np.where(diff[lower_bound:upper_bound] > threshold * np.max(diff))[0][0] + lower_bound
    except:
        # use trigger position when nothing found
        onset = -1-window_size
    try:
        onset_2 = np.argmax(diff[lower_bound:upper_bound]) + lower_bound
    except:
        onset_2 = -10000-window_size
    return onset + window_size, diff, onset_2 + window_size, lower_bound

def get_onset_3(waveform,window_size=100, threshold=[0.1], statistics=S_6, origin_sample=0):
    # Origin sample è il "tempo origine" dell'evento. Evito che trovo un segnale precedente!
    # BOUND on statistics, not on waveforms
    # Number of threshold arbitrary

    # get hos, here we use S 4
    hos = get_hos(waveform, window_size, statistics)
    # smooth the S4
    hos = np.convolve(hos, np.ones(3)/3, mode='valid')
    # get first derivative
    diff = np.diff(hos)
    # narrow the search range to a region near the maximum
    pre_window = 500
    #lower_bound = np.argmax(np.abs(waveform)) - 200 - window_size
    try:
        lower_bound = np.argmax(np.abs(hos[origin_sample:])) - pre_window + origin_sample
    except:
        onsets = [-1 for i in range(len(threshold))]
        lower_bound=-1
        onset_2 = -1-window_size
        return onsets, diff, onset_2 + window_size, lower_bound
    upper_bound = np.argmax(np.abs(hos[origin_sample:])) + pre_window + origin_sample               # MODIFIED

    onsets = []
    for i in range(len(threshold)):
        try:
            # find the onset larger than 0.1 * maximum of diff
            onsets.append(np.where(diff[lower_bound:upper_bound] > threshold[i] * np.max(diff))[0][0] + lower_bound + window_size)
        except:
            # use trigger position when nothing found
            onsets.append(-1)

    try:
        onset_2 = np.argmax(diff[lower_bound:upper_bound]) + lower_bound
    except:
        onset_2 = -10000-window_size
    return onsets, diff, onset_2 + window_size, lower_bound

def get_onset_4(waveform,window_size=100, threshold=[0.1], statistics=S_6, origin_sample=0, sampling_rate=200):
    # Origin sample è il "tempo origine" dell'evento. Evito che trovo un segnale precedente!
    # BOUND not on max of waveforms, not on max of statistics but constrain to be near the origin time
    # Number of threshold arbitrary
    # search intorno al massimo della statistica e NON al valore assoluto? 

    # get hos, here we use S 4
    hos = get_hos(waveform, window_size, statistics)
    # smooth the S4
    hos = np.convolve(hos, np.ones(3)/3, mode='valid')
    # get first derivative
    diff = np.diff(hos)
    # narrow the search range to a region near the maximum
    pre_window = 200 * sampling_rate//200


    lower_bound = np.argmax(hos) - pre_window
    if lower_bound < 0:
        lower_bound = 0
    upper_bound = lower_bound + pre_window +  window_size

    onsets = []
    for i in range(len(threshold)):
        try:
            # find the onset larger than 0.1 * maximum of diff
            onsets.append(np.where(diff[lower_bound:upper_bound] > threshold[i] * np.max(diff))[0][0] + lower_bound + window_size)
        except:
            # use trigger position when nothing found
            onsets.append(-100000)

    try:
        onset_2 = np.argmax(diff[lower_bound:upper_bound]) + lower_bound + window_size
    except:
        onset_2 = -100000
    return onsets, diff, onset_2, lower_bound

def cluster_div(picks: list, dmax=200, th=5/3, force_cut = False):
    """
    divisive hierarchical clustering with max distance by Gio to find clusters of picks
    
    dmax        :
    th          :
    force_cut   : force the division of the clusters with dimension > dmax
    picks       : list of numbers, the 
    sclust      : index of where a new cluster starts
    """
    # FIXME implement force_cut
    picks.sort()                            #
    pdiff = np.diff(picks)                  #
    sclust = np.where(pdiff>dmax / th)      # cluster size 2, not more than dmax / th (120)
    sclust = [0] + list(sclust[0]+1) + [len(picks)]

    for i in range(len(sclust)-1): # TODO ricorda sclust.sort() alla fine
        # Check all clusters with more than 2 elements and larger than dmax
        if (sclust[i+1] - sclust[i]) > 2 and (picks[sclust[i+1]-1] - picks[sclust[i]]) > dmax:
            3 # TODO non mi piace più il divisivo

def cluster_agg(picks: list, indexes=None, dmax=300, th=5):
    
    # agglomerative 1D hierarchical clustering with max distance by Gio, find clusters of picks

    # starting fromm all single elements representing a separate cluster,
    # agglomerate Reciprocal Nearest Neighbour,
    #     if diameter cluster < diam_max, accept new cluster (diam_max depends on numbers of points in the cluster)
    #     if distance of new link giving rise to new clusters < th * mean of other distances of original clusters, accept # FIXME RIVEDI!
    # Cycle
    # I End the cycle when no new cluster is born

    """ 
    Save the index of the starting and end of each cluster (sclust, eclust lists)
    
    i0  i1      i2            i3        i4             i5
    .   .      (.)<--------->(...)<--->(..)<--------->(.  .)
    
    Agglomerate i3 e i4 iif dist(3,4) < dist(2,3) & dist(3,4) < dist(4,5)
    Delete i4 from starting cluster list and i3 from ending cluster list
    """
    # picks.sort()
    picks = [picks[0]-3*dmax] + picks + [picks[-1]+3*dmax]

    sclust = [i for i in range(len(picks))]   # index representing the start of each cluster
    eclust = [i for i in range(len(picks))]   # index representing the end of each cluster
    agglomero = True
    while agglomero:
        agglomero = False
        sdel = []                              
        edel = []

        for i in range(1,len(sclust)-2):
            dl = picks[sclust[i]] - picks[eclust[i-1]]
            d0 = picks[sclust[i+1]] - picks[eclust[i]]
            dr = picks[sclust[i+2]] - picks[eclust[i+1]]        
            
            if d0<=dl and d0<=dr and d0<=dmax: # verify clusters are reciprocal Nearest Neighbour and New link not over maximum distance
                dnew = picks[eclust[i+1]] - picks[sclust[i]]    # size of the new candidate cluster
                Nnew = eclust[i+1] - sclust[i] + 1              # Number of component of the  new candidate

                if dnew < Nnew * dmax/8 + 3/8 * dmax:           # size not over the maximum size (225 for 3 points and 450 for 9 if dmax=300)
                    if Nnew == 2:
                        sdel.append(sclust[i+1])
                        edel.append(eclust[i])
                        agglomero = True
                    elif (d0 <= th * (picks[eclust[i+1]] - picks[sclust[i]] - d0) / (Nnew-2)) or d0 <=50:
                        sdel.append(sclust[i+1])
                        edel.append(eclust[i])
                        agglomero = True

        for i in sdel:
            sclust.remove(i)
        for i in edel:
            eclust.remove(i)
        #print(sclust,eclust, agglomero)

    for i in range(len(sclust)):
        sclust[i] -=1
    for i in range(len(eclust)):
        eclust[i] -=1

    return sclust[1:-1], eclust[1:-1]

def cluster_agg_max_distance(picks, dmax=300):
    # picks have to be a sorted list!
    pic_M = [ [i] for i in picks]
    Z = linkage(pic_M,"complete")       # "compute" the clustering procedure. returns the "rappresentation of the dendrogram"
    crit = Z[:, 2]
    flat_clusters = fcluster(Z, t=dmax, criterion='monocrit', monocrit=crit) # stops the clustering procedure based on criteria inside crit
    sclust=[0]
    eclust=[]
    for i in range(len(flat_clusters)):
        if i !=0:
            if flat_clusters[i-1] != flat_clusters[i]:
                sclust.append(i)
                eclust.append(i-1)
    eclust.append(len(flat_clusters)-1)

    return sclust, eclust

def accept_cluster(startclust:list,endclust:list):
    """
    startclust[i]: indice di dove inizia il cluster i-esimo
    endtclust[i]: indice di dove finisce il cluster i-esimo
    e.g. per cluster del vettore [1,2,3,50,51,100] con cluster pari a [[1,2,3], [50,51], 100], abbiamo:
            s = [0,3,5]
            e = [2,4,5]
    """
    # Una volta fatti i cluster, vedo il più popoloso (min 3), confronto con altri. 
    # Chiamo p1 e p2 le popolosità dei cluster maggiore e secondo maggiore. Affidabile se valgono tutte le seguenti:
        # 1) cluster maggiore comprende almeno metà dei pick (metà nel senso di //) o se ha 4 punti o di più
        # 2) p1 - p2 >= 2 (escludo caso 4,3)
        # 3) p1 >= 2 * p2 (escludo caso 6-4, 5-3)

    if len(startclust) != len(endclust):
        raise Exception("Len startclust does not match endclust")
    
    if len(startclust) == 1:
        return 0
    
    index_ok = -1
    size = np.array(endclust) - np.array(startclust) + 1 # diff contains the sizes of the clusters
    ssort = np.sort(size)
    if ((ssort[-1] > len(size)//2 ) or ssort[-1]>=4)  and (ssort[-1] - ssort[-2]) >=2 and (ssort[-1] >= 2*ssort[-2]):
        #Accetto!
        index_ok = np.argmax(size)

    return index_ok



"""
except:
onsets = [-1 for i in range(len(threshold))]
lower_bound=-1
onset_2 = -1-window_size
return onsets, diff, onset_2 + window_size, lower_bound"""