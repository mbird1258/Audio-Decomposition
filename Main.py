import os
import pickle
import soundfile as sf
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

# PARAMS
PointSpacing = 5
NoiseThreshold = 0.01
OverallNoiseThreshold = 1e-6
NoteNoiseThreshold = 0.02
CullThreshold = 0

EnvelopeNoiseThreshold = (5e-2, 1e-6)
PeakThresh = 0.25
SliceWindow = 5
SliceOverlap = 0.5
SliceInterval = 441
EnvelopeTuneIterations = 100

WhiteList = ['Piano', 'Flute', 'Oboe', 'Clarinet', 'Basoon', 'Horn', 'Trumpet', 'Trombone', 'Tuba', 'Xylophone', 'Cymbals', 'Accordion', 'Violin', 'Viola', 'Cello', 'Bass']
BlackList = []
MaxInstruments = 10

Playback = True

Notes = [                                                          'A0', 'Bb0', 'B0', 
         'C1', 'Db1', 'D1', 'Eb1', 'E1', 'F1', 'Gb1', 'G1', 'Ab1', 'A1', 'Bb1', 'B1', 
         'C2', 'Db2', 'D2', 'Eb2', 'E2', 'F2', 'Gb2', 'G2', 'Ab2', 'A2', 'Bb2', 'B2', 
         'C3', 'Db3', 'D3', 'Eb3', 'E3', 'F3', 'Gb3', 'G3', 'Ab3', 'A3', 'Bb3', 'B3', 
         'C4', 'Db4', 'D4', 'Eb4', 'E4', 'F4', 'Gb4', 'G4', 'Ab4', 'A4', 'Bb4', 'B4', 
         'C5', 'Db5', 'D5', 'Eb5', 'E5', 'F5', 'Gb5', 'G5', 'Ab5', 'A5', 'Bb5', 'B5', 
         'C6', 'Db6', 'D6', 'Eb6', 'E6', 'F6', 'Gb6', 'G6', 'Ab6', 'A6', 'Bb6', 'B6', 
         'C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7', 
         'C8', 'Db8', 'D8', 'Eb8', 'E8']

BaseDir = "/Users/matthewbird/Documents/Python Code/Song Decomposition/"
InstDir = os.listdir(BaseDir + "InstrumentData/")
InDir = os.listdir(BaseDir + "In/")

if '.DS_Store' in InstDir:
  InstDir.remove('.DS_Store')

if '.DS_Store' in InDir:
  InDir.remove('.DS_Store')


# Functions
def ReadInstruments():
    out = [[] for _ in range(6)]
    for file in InstDir:
        filename = os.fsdecode(file)
        
        if WhiteList:
            if not any(InstrumentName in filename for InstrumentName in WhiteList):
                continue
        else:
            if any(InstrumentName in filename for InstrumentName in BlackList):
                continue
        
        if not 'ff' in filename:
            continue

        name = f"{BaseDir}InstrumentData/{filename}"
        file = open(name, 'rb')
        x, y, keypoints, types, envelopes = pickle.loads(file.read())
        
        y = np.array(y)
        y[:5] = 0
        y[y < NoiseThreshold * np.max(y)] = 0
        envelopes = envelopes[::SliceInterval]

        for i, val in enumerate((filename, x, y, keypoints, types, envelopes)):
            out[i].append(val)
        file.close()
    
    for i in range(len(out)-1): 
        out[i] = np.array(out[i])
    out[-1] = np.array(out[-1], dtype=object)
    return out
instruments, frequencies, coefficients, keypoints, types, envelopes = ReadInstruments()

def GetSpectrogram(file):
    data, samplerate = sf.read(file)
    
    # If stereo, take one channel
    if data.ndim > 1:
        data = data[:, 0]

    # data = data[int(len(data) * 5/10) : int(len(data) * 7/10)] #

    # Compute the spectrogram
    NFFT = int(np.round(samplerate / PointSpacing))  # Number of points in each segment # Bigger = More resolution frequency axis,
    noverlap = NFFT // 2  # Number of overlapping points
    
    return data, samplerate, signal.spectrogram(data, fs=samplerate, nperseg=NFFT, noverlap=noverlap)

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

def GetEnvelope(data, cutoff, samplerate):
    chunk = int(len(data)/200)
    if len(cutoff) == 1:
        data = signal.datafilt(signal.butter(5, cutoff[0], fs=samplerate, btype='low', analog=False, output='data'), data)
    data = np.abs(data)

    DataSliced = np.pad(data, chunk*15)[::SliceInterval]
    chunk = int(chunk*len(DataSliced)/len(data))
    _, labs = hl_envelopes_idx(DataSliced,dmin=chunk,dmax=chunk,split=False)

    for i in range(EnvelopeTuneIterations):
        print(i, end = "\r")
        absinterp = np.interp(np.arange(len(DataSliced)), labs, DataSliced[labs])
        
        ind = np.argmax(DataSliced - absinterp)
        if np.abs(DataSliced[ind]) <= absinterp[ind]:
            break

        labs = np.sort(np.append(labs, ind))
    return absinterp, chunk

def GetEnvelopeCosts(data, start):
    out = []

    for ind, note in enumerate(Notes):
        cutoff = (np.power(2, (ind-1)/12)*27.5, np.power(2, (ind+1)/12)*27.5)
        sos = np.abs(signal.sosfilt(signal.butter(5, cutoff, fs=samplerate, btype="bandpass", analog=False, output='sos'), data))

        sos[sos <= EnvelopeNoiseThreshold[1] * np.max(data)] = 0

        if np.all(sos <= EnvelopeNoiseThreshold[0] * np.max(data)):
            continue
        
        sos, chunk = GetEnvelope(sos, cutoff, samplerate)
        
        fp = signal.convolve(sos, [1]*chunk*8, mode='same')[1:] - signal.convolve(sos, [1]*chunk*8, mode='same')[:-1]
        fp[fp < 0] = 0
        bounds = np.nonzero(fp <= np.max(fp)*0.01)[0]
        bounds = bounds[np.nonzero(~(bounds[1:] - bounds[:-1] == 1))[0]]

        peaks = []
        for ind1, ind2 in zip(bounds[:-1], bounds[1:]):
            peaks.append(np.argmax(fp[ind1:ind2+1])+ind1)
        peaks.append(np.argmax(fp[bounds[-1]:])+bounds[-1])
        peaks = np.array(peaks) + chunk*4
        
        for peak in peaks:
            if sos[peak] < np.max(sos[max(0, peak-samplerate):min(len(sos), peak+samplerate)])*PeakThresh:
                peaks = peaks[peaks != peak]

        print(f"      \n{note}")
        for InstInd, (instrument, keypoints, type, envelope) in enumerate(zip(InstanceInstruments, InstanceKeypoints, InstanceTypes, InstanceEnvelopes)):
            if not f".{note.lower()}." in instrument.lower():
                continue

            A, B, C, D = np.floor(keypoints/SliceInterval).astype(int)
            
            print(f"\t{instrument}")
            
            sustain = envelope[int(B)+1:int(C)]

            attack = envelope[int(A):int(B)+1]
            AttackIndexes = []
            for ind1, ind2 in zip(np.append(0, peaks[:-1]), peaks):
                corr = signal.correlate(sos[ind1:ind2+1+len(attack)], (attack-np.mean(attack))/np.std(attack), mode='valid')
                AttackIndexes.append(np.argmax(corr) + ind1)

            AttackIndexes = np.array(AttackIndexes)
            WaveA, WaveB = AttackIndexes, AttackIndexes + int(B-A)
            
            match type[0]:
                case "AS":
                    WaveC = []
                    for ind1, ind2 in zip(peaks, np.append(peaks[1:], len(sos)-1)):
                        WaveC.append(np.argmin(sos[ind1:ind2+1])+ind1)
                    WaveD = WaveC = np.array(WaveC)

                case "ASR":
                    release = envelope[int(C):int(D)+1]
                    ReleaseIndexes = []
                    for ind1, ind2 in zip(peaks, np.append(peaks[1:], len(sos)-1)):
                        corr = signal.correlate(sos[ind1:ind2+1+len(release)], (release-np.mean(release))/np.std(release), mode='valid')
                        ReleaseIndexes.append(np.argmax(corr) + ind1)
                    ReleaseIndexes = np.array(ReleaseIndexes)
                    
                    WaveC, WaveD = ReleaseIndexes, ReleaseIndexes + int(D-C)

            for WaveInd, (a, b, c, d) in enumerate(zip(WaveA, WaveB, WaveC, WaveD)):
                start = np.clip(b+chunk, 0, len(sos)-1)
                end = np.clip(c-chunk, 0, len(sos)-1)
                
                if start >= end:
                    start = np.clip(b, 0, len(sos)-1)
                    end = np.clip(c, 0, len(sos)-1)
                
                if start >= end:
                    continue

                r = (sos[end]/sos[start])**(1/(end-start))
                MSE = np.mean(np.power(sos[start:end+1] - r**np.arange(start, end+1), 2))
                if 1-r > 0 and MSE < 0.5: # Static
                    if type[1] != "Static":
                        continue
                else: # Dynamic
                    if type[1] != "Dynamic":
                        continue
                
                if type[1] == "Static":
                    WaveAttack = sos[max(a,0):b+1]
                    WaveSustain = sos[b+1:c+1]
                    if type[0] == "ASR": WaveRelease = sos[c+1:min(len(sos),d+1)]
                    
                    if len(WaveSustain) == 0:
                        continue

                    if len(WaveSustain) > len(sustain):
                        WaveSustain = WaveSustain[:len(sustain)]
                    
                    norm = lambda x: (x-np.mean(x))/np.std(x)
                    if type[0] == "ASR":
                        A = np.concatenate((norm(WaveAttack), norm(WaveSustain[:samplerate]), norm(WaveRelease[:int(samplerate*0.5)])), axis=None)
                        Y = np.concatenate((norm(attack[-len(WaveAttack):]), norm(sustain[:len(WaveSustain[:samplerate])]), norm(release[:len(WaveRelease[:int(samplerate*0.5)])])), axis=None)
                        out.append([ind, InstInd, start+a-chunk*15, start+d+1-chunk*15, 1/np.mean(np.power(A - Y, 2))])
                    else:
                        A = np.concatenate((norm(WaveAttack), norm(WaveSustain[:-int(samplerate*0.1)][:samplerate])), axis=None)
                        Y = np.concatenate((norm(attack[-len(WaveAttack):]), norm(sustain[:len(WaveSustain[:-int(samplerate*0.1)][:samplerate])])), axis=None)
                        out.append([ind, InstInd, start+a-chunk*15, start+c+1-chunk*15, 1/np.mean(np.power(A - Y, 2))])

                else:
                    WaveAttack = sos[max(a,0):b+1]
                    if type[0] == "ASR": WaveRelease = sos[c+1:min(len(sos),d+1)] 

                    if len(sos[b+1:c]) == 0:
                        continue

                    norm = lambda x: (x-np.mean(x))/np.std(x)
                    if type[0] == "ASR":
                        A = np.concatenate((norm(WaveAttack), norm(WaveRelease[:int(samplerate*0.5)])), axis=None)
                        Y = np.concatenate((norm(attack[-len(WaveAttack):]), norm(release[:len(WaveRelease[:int(samplerate*0.5)])])), axis=None)
                        out.append([ind, InstInd, start+a-chunk*15, start+d+1-chunk*15, 1/np.mean(np.power(A - Y, 2))])
                    else:
                        A = norm(WaveAttack)
                        Y = norm(attack[-len(WaveAttack):])
                        out.append([ind, InstInd, start+a-chunk*15, start+c+1-chunk*15, 1/np.mean(np.power(A - Y, 2))])

    return out # [[note index, instrument index, start time, end time, value]]

def FTScoring(tInd, transform):
    matrix = InstanceCoefficients @ np.concatenate((InstanceCoefficients.T, transform[:, np.newaxis]), axis = 1)
    out = magnitude = np.linalg.solve(matrix[:, :-1], matrix[:, -1:]).flatten()
    mask = np.ones(len(magnitude), np.bool_)
    
    CulledInstruments = np.array([], dtype = int)
    while not np.all(out*np.average(InstanceCoefficients[mask], axis = 1) >= np.max(out*np.average(InstanceCoefficients[mask], axis = 1)) * CullThreshold):
        CulledInstruments = np.union1d(np.nonzero(magnitude <= 0)[0], np.searchsorted(np.cumsum(mask)-1, np.argmin(out*np.average(InstanceCoefficients[mask], axis = 1)))) # np.union1d(CulledInstruments, np.argmin(magnitude))
        matrix = np.delete(InstanceCoefficients, CulledInstruments, 0) @ np.concatenate((np.delete(InstanceCoefficients, CulledInstruments, 0).T, transform[:, np.newaxis]), axis = 1)
        out = np.linalg.solve(matrix[:, :-1], matrix[:, -1:]).flatten()

        magnitude = np.zeros(InstanceInstruments.shape[0])
        mask = np.ones(len(magnitude), np.bool_)
        mask[CulledInstruments] = 0
        magnitude[mask] = out
    
    # plt.figure(figsize=(15, 2))
    # plt.plot(transform)
    # plt.plot(np.sum(InstanceCoefficients*magnitude[:, np.newaxis], axis = 0))
    # plt.show()

    return magnitude

for file in InDir:
    filename = os.fsdecode(file)
    name = f"{BaseDir}PlayBack/{filename.rsplit('.', 1)[0]}.pkl"

    data, samplerate, (f, t, Sxx) = GetSpectrogram(f"{BaseDir}In/{filename}")
    f = f[:int(np.round(22000/PointSpacing))]
    Sxx = Sxx[:int(np.round(22000/PointSpacing))]
    Sxx[:5] = 0

    # cull instruments
    dictionary = {}
    print("\n"*2, end = "")
    for ind, (instrument, coefficient) in enumerate(zip(instruments, coefficients)):
        print("\033[F\033[K",  end = "")
        print(f"{ind}/{len(instruments)}")

        cost = coefficient[:, np.newaxis] - Sxx
        cost = np.sum(cost * (cost > 0))

        key = instrument.split(".")[0]
        if not key in dictionary:
            dictionary[key] = [cost, [instrument]]
        else:
            dictionary[key][0] = min(dictionary[key][0], cost)
            dictionary[key][1].append(instrument)
    dictionary = np.array(list(dictionary.items()), dtype = object)
    InstanceInstruments = np.array([i[1] for i in dictionary[:, 1]], dtype = object)[np.argsort([i[0] for i in dictionary[:, 1]])[:MaxInstruments]]
    InstanceInstruments = np.array([i for j in InstanceInstruments for i in j])

    InstanceCoefficients = coefficients[np.isin(instruments, InstanceInstruments)]
    InstanceCoefficients[InstanceCoefficients <= np.max(InstanceCoefficients, axis = 1)[:, np.newaxis]*NoiseThreshold] = 0

    InstanceKeypoints = keypoints[np.isin(instruments, InstanceInstruments)]
    InstanceTypes = types[np.isin(instruments, InstanceInstruments)]
    InstanceEnvelopes = envelopes[np.isin(instruments, InstanceInstruments)]
    InstanceInstruments = instruments[np.isin(instruments, InstanceInstruments)]

    # get notes and magnitudes of instruments at each time index
    notes = []
    magnitudes = []

    for tInd, transform in enumerate(Sxx.T):
        print("\033[F\033[K",  end = "")
        print(f"iteration:   {tInd}/{Sxx.shape[1]}")

        note = np.nonzero(transform >= NoteNoiseThreshold * np.max(transform))[0] * PointSpacing

        transform[transform < max(NoiseThreshold * np.max(transform), OverallNoiseThreshold)] = 0
        magnitude = FTScoring(tInd, transform)
        
        notes.append(note)
        magnitudes.append(magnitude)
    magnitudes = np.array(magnitudes)

    # get envelope costs of each instrument
    out1 = []
    out2 = []

    WindowSize = samplerate*SliceWindow
    SplitData = np.pad(data, (0,WindowSize-((len(data)-1)%WindowSize+1))).reshape(-1, WindowSize)
    for ind, slice in enumerate(SplitData):
        print(f"1 - {ind}/{len(SplitData)} | 2 - 0/{len(SplitData)}")

        out1.extend(GetEnvelopeCosts(slice, ind*WindowSize))

    overlap = int(WindowSize*SliceOverlap/SliceInterval)
    for ind, slice in enumerate(np.pad(SplitData.reshape(-1)[overlap:], (0,overlap)).reshape(-1, WindowSize)):
        print(f"1 - {len(SplitData)}/{len(SplitData)} | 2 - {ind}/{len(SplitData)}")
        
        out2.extend(GetEnvelopeCosts(slice, ind*WindowSize+overlap))

    out1 = np.array(out1)
    out2 = np.array(out2)

    out1[:, [2,3]] *= SliceInterval
    out2[:, [2,3]] *= SliceInterval
    
    # save to file
    temp = magnitudes.copy()
    for tInd, magnitude in enumerate(temp):
        time = tInd*(t[1]-t[0])
        
        tOut1 = out1[np.logical_and(time >= out1[:, 2], time <= out1[:, 3])]
        tOut2 = out2[np.logical_and(time >= out2[:, 2], time <= out2[:, 3])]

        instOut1 = False
        for InstInd, InstMag in enumerate(magnitude):
            if InstMag == 0:
                continue

            instOut1 = tOut1[InstInd == tOut1[:, 1]][:, 4]
            instOut2 = tOut2[InstInd == tOut2[:, 1]][:, 4]

            if len(instOut1) == 0 and len(instOut2) == 0:
                continue

            temp[tInd][InstInd] = InstMag * np.max(np.append(instOut1.flatten(), instOut2.flatten()))

    file = open(name, 'wb')
    file.write(pickle.dumps([t[1]-t[0], InstanceInstruments, temp, notes]))
    file.close()

    '''
    https://4kdownload.to/envn/youtube-wav-downloader
    '''