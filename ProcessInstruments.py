import numpy as np
from scipy import signal
import soundfile as sf
import os
import pickle
import matplotlib.pyplot as plt

# PARAMS
PointSpacing = 5
BaseDir = os.path.dirname(os.path.realpath(__file__)) + "/"
SecondDerivThreshold = 0.02
dilation = (3,5)

Notes = [                                                          'A0', 'Bb0', 'B0', 
         'C1', 'Db1', 'D1', 'Eb1', 'E1', 'F1', 'Gb1', 'G1', 'Ab1', 'A1', 'Bb1', 'B1', 
         'C2', 'Db2', 'D2', 'Eb2', 'E2', 'F2', 'Gb2', 'G2', 'Ab2', 'A2', 'Bb2', 'B2', 
         'C3', 'Db3', 'D3', 'Eb3', 'E3', 'F3', 'Gb3', 'G3', 'Ab3', 'A3', 'Bb3', 'B3', 
         'C4', 'Db4', 'D4', 'Eb4', 'E4', 'F4', 'Gb4', 'G4', 'Ab4', 'A4', 'Bb4', 'B4', 
         'C5', 'Db5', 'D5', 'Eb5', 'E5', 'F5', 'Gb5', 'G5', 'Ab5', 'A5', 'Bb5', 'B5', 
         'C6', 'Db6', 'D6', 'Eb6', 'E6', 'F6', 'Gb6', 'G6', 'Ab6', 'A6', 'Bb6', 'B6', 
         'C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7', 
         'C8', 'Db8', 'D8', 'Eb8', 'E8']

frequencies = np.power(2, np.arange(len(Notes))/12)*27.5

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) >= 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) <= 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def GetPoints(data, samplerate):
    # Compute the FFT
    N = len(data)
    T = 1.0 / samplerate
    yf = np.fft.fft(data)
    xf = np.fft.fftfreq(N, T)[:N // 2]

    # Compute the magnitude spectrum
    magnitude = 2.0 / N * np.abs(yf[:N // 2])

    # Get points of interest
    HzToInd = N/samplerate

    out = [np.arange(0, 22000, PointSpacing), []]
    for Hz in out[0]:
        low = np.floor((Hz-PointSpacing/2)*HzToInd).astype(int)
        high = np.ceil((Hz+PointSpacing/2)*HzToInd).astype(int)
        out[1].append(np.max(magnitude[np.arange(low,high)]))

    return out

def GetEnvelope(data, samplerate, cutoff):
    chunk = int(len(data)/200)
    # data[44100:] = 0
    if len(cutoff) == 1:
        cutoff = (np.power(2, (cutoff[0]-1)/12)*27.5, np.power(2, (cutoff[0]+1)/12)*27.5)
        data = signal.sosfilt(signal.butter(5, cutoff, fs=samplerate, btype="bandpass", analog=False, output='sos'), data)
    data = np.abs(data)

    data = np.pad(data, chunk*15)
    _, labs = hl_envelopes_idx(data,dmin=chunk,dmax=chunk,split=False)

    A, D = np.nonzero(data > 1e-2*np.max(data))[0][[0, -1]]
    labs = np.sort(np.append(labs, [A, D]))

    for i in range(1000):
        print(i, end = "\r")
        absinterp = np.interp(np.arange(len(data)), labs, data[labs])
        
        ind = np.argmax(data - absinterp)
        if np.abs(data[ind]) <= absinterp[ind]:
            break

        labs = np.sort(np.append(labs, ind))
    
    fp = absinterp[1:] - absinterp[:-1]
    temp = signal.convolve(fp, [1]*chunk*10, mode='same')
    fp = signal.convolve(fp, [1]*chunk*dilation[0], mode='same')

    fpp = fp[1:]-fp[:-1]
    absfpp = signal.convolve(np.abs(fpp), [1]*chunk*dilation[1], mode='same')
    fpp = signal.convolve(fpp, [1]*chunk*dilation[1], mode='same')
    
    try:
        bound1, bound2 = np.array(signal.find_peaks(np.abs(temp[A:D+1]), height = np.max(np.abs(temp[A:D+1]))/5)[0])[[0, -1]] + A
    except:
        try:
            bound1, bound2 = np.array(signal.find_peaks(np.abs(temp[A:D+1]))[0])[[0, -1]] + A
        except:
            bound1, bound2 = A, D

    temp = fpp[bound1:].copy()
    temp[temp >= 0] = 0
    peaks, _ = signal.find_peaks(-temp, height = np.max(-temp)/20)
    if len(peaks) == 0: peaks = [A]
    B = np.nonzero(absfpp[peaks[0]+bound1:] <= np.max(np.abs(absfpp))*SecondDerivThreshold)[0][0]+peaks[0]+bound1
    B = min(np.nonzero(fp[bound1:] <= 0)[0][0] + bound1, B)

    B = np.nonzero((absinterp[1:B+1] - absinterp[:B]) >= 0)[0][-1]
    
    temp = fpp[:bound2+1].copy()
    temp[temp >= 0] = 0
    peaks, _ = signal.find_peaks(-temp)
    
    C = np.nonzero(absfpp[:peaks[-1]+1] <= np.max(np.abs(absfpp))*SecondDerivThreshold)[0][-1]
    C = max(np.nonzero(fp[:bound2+1] >= 0)[0][-1], C)

    type = [None, None]
    if C <= B: # AS
        B = min(B, C)
        C = D
        D = np.inf
        type[0] = "AS"
    
    else:
        ReleaseFactor = ((absinterp[D]-absinterp[C])/(D-C)) / ((absinterp[C]-absinterp[B])/(C-B))
        if ReleaseFactor >= 0 and ReleaseFactor <= 1: # AS
            B = min(B, C)
            C = D
            D = np.inf
            type[0] = "AS"
        else:
            type[0] = "ASR"
        
    r = (absinterp[C-chunk]/absinterp[B+chunk])**(1/(C-B-chunk*2))
    MSE = np.mean(np.power(absinterp[np.linspace(B+chunk, C-chunk, dtype=int)] - r**np.linspace(B+chunk, C-chunk, dtype=int), 2))
    if 1-r > 0 and MSE < 0.5: # Static
        type[1] = "Static"
    else: # Dynamic
        type[1] = "Dynamic"
    
    return [np.array([A,B,C,D])-chunk*15, type, absinterp[chunk*15:-chunk*15]]

def StoreData():
    for file in os.listdir(BaseDir + "InstrumentAudioFiles/"):
        filename = os.fsdecode(file)
        name = f"{BaseDir}InstrumentData/{filename.rsplit('.', 1)[0]}.pkl"
        if os.path.isfile(name):
            if os.path.getsize(name):
                print(f"{name} already exists, skipping")
                continue
        
        if not ".ff." in name:
            continue

        # if not "Violin" in name and not "Piano" in name:
        #     continue

        if not "Piano.ff.F3" in name:
            continue

        if filename.endswith(".aiff") or filename.endswith(".aif"):
            print(filename)
            data, samplerate = sf.read(f"{BaseDir}InstrumentAudioFiles/{filename}")
            cutoff = np.nonzero([note in filename for note in Notes])[0]

            # If stereo, take one channel
            if data.ndim > 1:
                data = data[:, 0]  # Use the first channel

            # try:
            #     temp = GetEnvelope(data, samplerate, cutoff)
            #     plt.plot(temp[2])
            #     plt.scatter(temp[0], temp[2][temp[0]])
            #     plt.show()
            # except:
            #     temp = GetEnvelope(data, samplerate, cutoff)
            #     plt.plot(temp[2])
            #     plt.scatter(temp[0][:-1], temp[2][temp[0][:-1]])
            #     plt.show()

            file = open(name, 'wb')
            file.write(pickle.dumps([*GetPoints(data, samplerate), *GetEnvelope(data, samplerate, cutoff)]))
            file.close()
        else:
            print("Error encountered: filename is", filename)

    print("    \nfin")

StoreData()