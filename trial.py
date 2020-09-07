#import music21
from music21 import converter, instrument, note, chord
import glob
import numpy
import scipy
import matplotlib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.utils import np_utils

notes = []
i=0
for file in glob.glob("chopin/*.mid"):
    i = i + 1
    parsed_file = converter.parse(file)

    print("Parsing file: " + str(i))

    segment = parsed_file.flat.notes

    for stuff in segment:
        if stuff.isNote:
            notes.append(str(stuff.pitch))
        elif stuff.isChord:
            l = ""
            for c in stuff.normalOrder:
                L = str(c)
                l = l + L

sequence_length = 20

    # get all pitch names
pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

    # create input sequences and the corresponding outputs
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)
n_vocab = len(set(notes))

    # reshape the input into a format compatible with LSTM layers
network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))