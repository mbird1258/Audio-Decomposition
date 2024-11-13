from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import pickle
import numpy as np
import colorsys
from datetime import datetime

ViewMode = 0

MagnitudeThresh = 0.2
CullByOverallNote = True

BaseDir = os.path.dirname(os.path.realpath(__file__))
PlayBackDir = os.listdir(os.path.join(BaseDir, "PlayBack"))
InstDir = os.listdir(os.path.join(BaseDir, "InstrumentData"))

InstrumentDescriptors = []
ExceptionDescriptors = []

YAxis = [                                                          'A0', 'Bb0', 'B0',
         'C1', 'Db1', 'D1', 'Eb1', 'E1', 'F1', 'Gb1', 'G1', 'Ab1', 'A1', 'Bb1', 'B1',
         'C2', 'Db2', 'D2', 'Eb2', 'E2', 'F2', 'Gb2', 'G2', 'Ab2', 'A2', 'Bb2', 'B2',
         'C3', 'Db3', 'D3', 'Eb3', 'E3', 'F3', 'Gb3', 'G3', 'Ab3', 'A3', 'Bb3', 'B3',
         'C4', 'Db4', 'D4', 'Eb4', 'E4', 'F4', 'Gb4', 'G4', 'Ab4', 'A4', 'Bb4', 'B4',
         'C5', 'Db5', 'D5', 'Eb5', 'E5', 'F5', 'Gb5', 'G5', 'Ab5', 'A5', 'Bb5', 'B5',
         'C6', 'Db6', 'D6', 'Eb6', 'E6', 'F6', 'Gb6', 'G6', 'Ab6', 'A6', 'Bb6', 'B6',
         'C7', 'Db7', 'D7', 'Eb7', 'E7', 'F7', 'Gb7', 'G7', 'Ab7', 'A7', 'Bb7', 'B7',
         'C8', 'Db8', 'D8', 'Eb8', 'E8']

frequencies = np.power(2, np.arange(len(YAxis))/12)*27.5

for file in PlayBackDir:
    if os.path.basename(file).startswith('.'):
        continue

    if(not file.endswith(".pkl")):
        continue

    filename = os.fsdecode(file)
    with open(os.path.join(BaseDir, "PlayBack", filename), 'rb') as file:
        TemporalResolution, instruments, magnitudes, notes = pickle.loads(file.read())

    N1 = 0
    N2 = 0
    for instrument in instruments:
        if not '.ff.' in instrument:
            continue

        prefix, affix = instrument.split('.ff.')

        if 'guitar' in prefix.lower():
            N1 += 1
            ExceptionDescriptors.append(f"{prefix}.{affix.split('.')[0]}.{affix.split('.')[1]}")
            YAxis.append(prefix)
            continue

        if affix[:4].lower() == 'mono' or affix[:5].lower() == 'stereo':
            N1 += 1
            ExceptionDescriptors.append(prefix)
            YAxis.append(prefix)
            continue

        if affix[:3].lower() == 'sul' and not f"{prefix}.{affix[:4]}" in InstrumentDescriptors:
            N2 += 1
            InstrumentDescriptors.append(f"{prefix}.{affix[:4]}")
            continue

        if not prefix in InstrumentDescriptors:
            N2 += 1
            InstrumentDescriptors.append(prefix)
            continue

    N = N1+N2
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    ExceptionDescriptors = dict(zip(ExceptionDescriptors, RGB_tuples[:N1]))
    InstrumentDescriptors = dict(zip(InstrumentDescriptors, RGB_tuples[N1:]))
    LegendDict = dict()

    plt.figure(figsize=(10,7))
    plt.yticks(np.arange(len(YAxis)), YAxis, size = 8)
    plt.xticks(np.arange(np.ceil(len(magnitudes) * TemporalResolution))/TemporalResolution, np.arange(np.ceil(len(magnitudes) * TemporalResolution)))
    plt.gca().xaxis.set_major_locator(MultipleLocator(1/TemporalResolution))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(color='#bbbbbb')
    plt.grid(True, 'minor', color='#eeeeee')
    plt.ylim(0, len(YAxis)+1)

    if CullByOverallNote:
        for ind, TimeNotes in enumerate(notes):
            notes[ind] = np.array(YAxis)[[(np.abs(note - frequencies)).argmin() for note in TimeNotes]]

    for ind, (instrument, magnitude) in enumerate(zip(instruments, magnitudes)):
        coefficients = None
        for file in InstDir:
            filename = os.fsdecode(file)
            if instrument in filename:
                name = os.path.join(BaseDir, "InstrumentData", filename)
                with open(name, 'rb') as file:
                    _, coefficients, _, _, _ = pickle.loads(file.read())
                break
        magnitudes[ind] = magnitude * np.average(coefficients)

    x = []
    y = []
    if CullByOverallNote:
        for tInd, TimeNotes in enumerate(notes):
            TimeY = [np.nonzero(np.array(YAxis) == note)[0][0] for note in TimeNotes]
            y.extend(TimeY)
            x.extend(np.repeat(tInd, len(TimeY)))
        LegendDict['possibilities marker'] = plt.scatter(x, y, color=[0.8, 0.8, 0.8, 1], s=50, marker='+', label='possibilities marker')

    if CullByOverallNote:
        for InstInd, (instrument, InstNotes) in enumerate(zip(instruments, magnitudes.T)):
            prefix, affix = instrument.split('.ff.')
            if prefix in ExceptionDescriptors:
                continue

            for tInd, TimeNotes in enumerate(notes):
                if not any([note in instrument for note in TimeNotes]):
                    magnitudes[tInd][InstInd] = 0

    lim = np.max(magnitudes, axis = 1)*MagnitudeThresh

    for instrument, magnitude in zip(instruments, magnitudes.T):
        if np.sum(magnitude) == 0:
            continue

        if not '.ff.' in instrument:
            continue

        prefix, affix = instrument.split('.ff.')

        if 'pizz' in prefix.lower() or 'arco' in prefix.lower():
            affix = affix.split('.',1)[1]

        if prefix in ExceptionDescriptors:
            y = np.nonzero(np.array(YAxis) == prefix)[0][0]
            C = np.ones((len(magnitude), 4))
            C[:, [0,1,2]] *= np.array(ExceptionDescriptors[f"{prefix}.{affix.split('.')[0]}.{affix.split('.')[1]}"]) if 'guitar' in prefix.lower() else np.array(ExceptionDescriptors[prefix])
            C[:, 3] *= np.clip(magnitude/lim, 0, 1)
            LegendDict[prefix] = plt.scatter(np.arange(len(magnitude))[magnitude != 0], np.repeat(y, len(magnitude))[magnitude != 0], c=C[magnitude != 0], s=100, marker='_', label=prefix)
            continue

        if f"{prefix}.{affix[:4]}" in InstrumentDescriptors:
            y = np.nonzero(np.array(YAxis) == affix.split('.')[1])[0][0]
            C = np.ones((len(magnitude), 4))
            C[:, [0,1,2]] *= np.array(InstrumentDescriptors[f"{prefix}.{affix[:4]}"])
            C[:, 3] *= np.clip(magnitude/lim, 0, 1)
            LegendDict[f"{prefix}.{affix[:4]}"] = plt.scatter(np.arange(len(magnitude))[magnitude != 0], np.repeat(y, len(magnitude))[magnitude != 0], c=C[magnitude != 0], s=100, marker='_', label=f"{prefix}.{affix[:4]}")
            continue

        if prefix in InstrumentDescriptors:
            y = np.nonzero(np.array(YAxis) == affix.split('.')[0])[0][0]
            C = np.ones((len(magnitude), 4))
            C[:, [0,1,2]] *= np.array(InstrumentDescriptors[prefix])
            C[:, 3] *= np.clip(magnitude/lim, 0, 1)
            LegendDict[prefix] = plt.scatter(np.arange(len(magnitude))[magnitude != 0], np.repeat(y, len(magnitude))[magnitude != 0], c=C[magnitude != 0], s=100, marker='_', label=prefix)
            continue

    # plt.xlim(0, 50)
    # plt.show()

    plt.gca().set_aspect('auto')
    plt.gca().set_axisbelow(True)

    match ViewMode:
        case 0:
            leg = plt.legend(handles=LegendDict.values(), loc='upper right')
            for lh in leg.legendHandles:
                lh.set_alpha(1)

            plt.ion()
            plt.waitforbuttonpress()
            t0 = datetime.now().timestamp()
            while True:
                x = datetime.now().timestamp()-t0
                if x/TemporalResolution > len(magnitudes):
                    break
                plt.xlim((x-2.5)/TemporalResolution, (x+2.5)/TemporalResolution)
                line = plt.plot([x/TemporalResolution, x/TemporalResolution], [0, len(YAxis)], c='red')
                plt.pause(1/30)
                line.pop(0).remove()

        case 1:
            x = 0
            while True:
                plt.xlim((x)/TemporalResolution, (x+1)/TemporalResolution)
                plt.waitforbuttonpress()

                if x/TemporalResolution > len(magnitudes):
                    break

                x += 0.1
