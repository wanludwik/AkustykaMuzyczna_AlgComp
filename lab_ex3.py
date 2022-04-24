#-*- coding:utf8 -*-

# ------------------------------------------------------------------------------------------------------
#   Sekcja importu bibliotek:
#
#   Fragment do modyfikacji w ramach ćwiczenia znajduje się na dole, na koncu niniejszego pliku.
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import scipy.io.wavfile as wav
import scipy.signal as signal
import os
import sys

import keras
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM



sys.path.append('../_utils')
from MIDI_IO import *


# ------------------------------------------------------------------------------------------------------
#   Odczyt przykładów z folderu wskazanego w zmiennej examples_directory

examples_directory = "music_examples"
examples = []

for file in os.listdir(examples_directory):
    if file.endswith(".mid"):
        m = MidiParser(examples_directory+"/"+file, 6)
        examples.append(m.output_piano_roll_representation)

# funkcja przygotowująca wzorce uczące dla algorytmu trenowania sieci LSTM
def generate_sets(data_array, length_of_window):
    length_of_sequence = data_array.shape[0]

    train_X = []
    train_Y = []

    for i in range(0,length_of_sequence-length_of_window-1):
        train_X.append(data_array[i:i+length_of_window,:])
        train_Y.append(data_array[i+length_of_window+1,:])

    return (np.array(train_X), np.array(train_Y))

# odzyskiwanie wektora w formie one-hot z wektora zwróconego przez sieć neuronową
def retrieve_one_hot(predicted_vector):
    output_vector = np.zeros(predicted_vector.shape)
    output_vector[0,np.argmax(predicted_vector)] = 1
    return output_vector.astype(int).tolist()[0]

# -------------------------------------------------------------
def generate_model(examples, learning_epochs_number, number_of_previous_samples):

    # przygotowanie wektorów zawierających dane treningowe
    train_X = []
    train_Y = []

    for example in examples:
        number_of_inputs = len(example[0])
        example_train_X, example_train_Y = generate_sets(np.array(example),number_of_previous_samples)
        train_X.append(example_train_X)
        train_Y.append(example_train_Y)

    train_X = np.vstack(train_X)
    train_Y = np.vstack(train_Y)

    # definicja struktury sieci neuronowej
    model = Sequential()
    model.add(LSTM(160, input_shape=(number_of_previous_samples ,number_of_inputs), return_sequences=True, activation='elu'))
    model.add(LSTM(160, activation='elu'))
    model.add(Dense(number_of_inputs))

    # kompilacja sieci oraz wytrenowanie jej
    optimizer = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    model.fit(train_X,train_Y,epochs=learning_epochs_number,batch_size=512)

    return model

# -------------------------------------------------------------
def generate_melody(examples, model, number_of_previous_samples, melody_length):

    # initialization of previous output state memory list (from the first example)
    index_of_example_used_for_initialization = np.random.randint(0,len(examples))

    previous_outputs_memory = []
    for i in range(0,number_of_previous_samples):
        previous_outputs_memory.append(examples[index_of_example_used_for_initialization][i])

    # generation stage
    output_vectors = []
    for i in range(0,melody_length):
        predicted_sample = retrieve_one_hot(model.predict(np.array([np.array(previous_outputs_memory)])))

        previous_outputs_memory.pop(0)
        previous_outputs_memory.append(predicted_sample)

        output_vectors.append(predicted_sample)

    return output_vectors

# ------------------------------------------------------------------------------------------------------
#                      Nastawy pracy algorytmu (TUTAJ EDYTOWAC)
# ------------------------------------------------------------------------------------------------------

print("Zadanie 3: generowanie melodii za pomoca sieci neuronowej LSTM")

# liczba poprzedzających wektorów one hot opisujących melodię, które sieć uwzględnia w trakcie generowania
# nowych linii melodycznych
number_of_previous_samples  = 25

# długość melodii generowanej przez sieć nuronową (podawana w liczebności wygenerowanych wektorów one hot)
length_of_melody            = 250

# liczba epok (iteracji algorytmu uczenia) treningu sieci neuronowej
learning_epochs_number      = 50

# polecenie wygenerowania nowego modelu sieci neuronowej
neural_network_model        = generate_model(examples,learning_epochs_number,number_of_previous_samples)

# polecenie zapisu utworzonego modelu sieci neuronowej
neural_network_model.save("last_network_model.model")

# nazwa pliku, z którego ewentualnie model sieci neuronowej ma być odczytany
# name_of_file_with_model_to_be_imported  = "last_network_model.model"
# polecenie odczytu utworzonego modelu sieci neuronowej
# neural_network_model                    = load_model(name_of_file_with_model_to_be_imported)

# generowanie N melodii
N = 10
for n in range(N):
    print('-------------------------------\n')
    print('generating melody %3i out of %i'%(n+1,N))
    # wygenrowanie nowej melodii przy wykorzystaniu modelu zapisanego w zmiennej neural_network_model
    generated_melody          = generate_melody(examples, neural_network_model,number_of_previous_samples,length_of_melody)

    # przejdź z zapisu w formie wektoów "one hot" na zapis w formie nut
    melody_descriptor = condensate_pianoroll(generated_melody, 60, 2.5)

    # wypisz wygenerowaną sekwencję nut na ekranie
    for note in melody_descriptor: print(note)

    # zapisz wygenerowaną sekwencję nut do pliku midi
    output_file_name            = 'output_examples/ann_composition_%s.mid'%str(n).zfill(2)
    save_melody_to_midi(melody_descriptor, output_file_name)