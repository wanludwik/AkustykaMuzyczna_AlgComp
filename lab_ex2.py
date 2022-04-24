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

sys.path.append('../_utils')
from MIDI_IO import *

# -----------------------------------
# logika generowania gramatyk stochastycznych

# opis obiektu reprezentującego pojedynczy stan modelu
class MarkovChainStateDescriptor:
    def __init__(self, previous_states):
        # previous notes that were in melody
        self.previous_states            = previous_states
        # probability distribution for the next note(interval)
        self.frequency_distribution   = []

    def increment_frequency_for(self, note_id):
        frequency_was_incremented = False
        for row in self.frequency_distribution:
            if row[0] == note_id:
                row[1] += 1
                frequency_was_incremented = True
        if not frequency_was_incremented:
            self.frequency_distribution.append([note_id,1])

    def state_keys_are_equal(self, state_keys):
        for i, key in enumerate(state_keys):
            if key != self.previous_states[i]:
                return False
        return True

    def get_probability_distribution(self):
        number_of_cases = np.sum(np.array(self.frequency_distribution)[:,1])
        key_values      = np.array(self.frequency_distribution)[:,0]
        probabilities   = np.array(self.frequency_distribution)[:,1]/float(number_of_cases)
        distribution    = []
        for i in range(0,len(key_values)):
            distribution.append([key_values[i], probabilities[i]])
        return list(sorted(distribution, key=lambda x:x[1], reverse=True))

    def get_cumulative_distribution(self):
        cumulative_distribution = self.get_probability_distribution()
        for i in range(1,len(cumulative_distribution)):
            key, probability = cumulative_distribution[i]
            cumulative_distribution[i][1] = probability + cumulative_distribution[i-1][1]
        return cumulative_distribution

    def get_next_value(self):
        cumulative_distribution = self.get_cumulative_distribution()
        random_number           = np.random.uniform(0,1,1)[0]

        for value, threshold in cumulative_distribution:
            if random_number > threshold:
                continue
            else:
                return value

    def get_probability_for(self, value):
        probability_distribution = self.get_probability_distribution()
        for key, probability in probability_distribution:
            if key == value:
                return probability
            else:
                return None

    def __str__(self):
        string_buffer = "["
        for number in self.previous_states:
            string_buffer += (str(number)+",")
        string_buffer+="] ---> \n"

        probability_distribution = self.get_probability_distribution()
        for key, probability in probability_distribution:
            string_buffer+=("\t"+str(key)+": "+str(probability)+"\n")

        string_buffer += "\n"
        return string_buffer

def find_markov_chain_state_descriptor(previous_notes, markov_chain_descriptor):
    for mc_state_descriptor in markov_chain_descriptor:
        if mc_state_descriptor.state_keys_are_equal(previous_notes):
            return mc_state_descriptor
    return None

def get_markov_chain_from_melody(melody_profile, markov_model_order):
    number_of_notes = len(melody_profile)
    markov_chain_descriptor = []

    for i in range(markov_model_order,number_of_notes-markov_model_order):
        previous_notes  = melody_profile[i-markov_model_order:i]
        current_note    = melody_profile[i]

        current_mc_state_descriptor = find_markov_chain_state_descriptor(previous_notes, markov_chain_descriptor)

        if current_mc_state_descriptor  is None:
            new_state_descriptor = MarkovChainStateDescriptor(previous_notes)
            new_state_descriptor.increment_frequency_for(current_note)
            markov_chain_descriptor.append(new_state_descriptor)
        else:
            current_mc_state_descriptor.increment_frequency_for(current_note)

    return markov_chain_descriptor

def draw_random_note():
    random_number = np.random.uniform(0,1)*24
    return np.ceil(random_number)-12

def generate_melody_from_chain(chain_descriptor, melody_length):
    markov_model_order     = len(chain_descriptor[0].previous_states)

    melody_profile = []

    random_index = int(np.floor(np.random.uniform(0, 1)*float(len(chain_descriptor))))
    melody_profile += chain_descriptor[random_index].previous_states

    i = markov_model_order
    while i < melody_length:
        previous_notes              = melody_profile[i-markov_model_order:i]
        current_state_descriptor    = find_markov_chain_state_descriptor(previous_notes,chain_descriptor)

        if current_state_descriptor is None:
            random_number   = np.random.uniform(0, 1)
            random_index    = int(np.floor(random_number*len(chain_descriptor)))
            melody_profile += chain_descriptor[random_index].previous_states
            i += (len(chain_descriptor[random_index].previous_states)-1)
        else:
            melody_profile.append(current_state_descriptor.get_next_value())
            i += 1

    return melody_profile

def calculate_intervals_from_melody(melody):
    intervals = []
    for i in range(1,len(melody)):
        intervals.append(melody[i] - melody[i-1])
    return intervals

def calculate_melody_from_intervals(intervals):
    melody = [0]
    for i in range(1,len(intervals)):
        melody.append(melody[i-1]+intervals[i])
    return melody

def markov_resynthesis(original_melody, chain_order):
    mc = get_markov_chain_from_melody(calculate_intervals_from_melody(original_melody),chain_order)

    rules_raport = ""
    rules_raport += ("number of rules: "+str(len(mc))+"\n")
    rules_raport += "\n"
    for mcs in mc:
        rules_raport += str(mcs)

    new_melody = calculate_melody_from_intervals(generate_melody_from_chain(mc,40))
    return (rules_raport, new_melody)


# -----------------------------------
# obsługa odczytu i przekomponowywania melodii

def read_melody(file_name):
    output_intervals = []

    m = MidiParser(file_name, 12)
    for i in range(0,len(m.output_note_object_representation)):
        current_pitch   = m.output_note_object_representation[i].pitch

        output_intervals.append(current_pitch)

    return output_intervals

def process_melodies(examples_directory, output_directory, model_order, number_of_generated_melodies=10, starting_note=60, note_duration=0.5):

    training_sequence = []

    print('Reading training examples...')
    for file in os.listdir(examples_directory):
        if file.endswith(".mid"):
            training_sequence = training_sequence + read_melody(examples_directory + "/" + file)

    save_grammar = True
    
    for i in range(0,number_of_generated_melodies):
        print('Generating melody number %3i out of %i'%(i+1,number_of_generated_melodies))
        rules_raport, resynthesized_melody = markov_resynthesis(training_sequence, model_order)

        output_notes = [Note(starting_note, note_duration)]
        for j in range(0, len(resynthesized_melody)):
            new_pitch = resynthesized_melody[j]+60

            allowed_range=2
            if new_pitch > 60+12*allowed_range: new_pitch=new_pitch-12*allowed_range
            if new_pitch < 60-12*allowed_range: new_pitch=new_pitch+12*allowed_range

            if new_pitch > 127: new_pitch = 127
            if new_pitch < 0: new_pitch=0

            output_notes.append(Note(new_pitch, note_duration))

        output_file_name = "markov_processed_melody_"+str(i).zfill(3)+".mid"
        save_melody_to_midi(output_notes, output_directory+"/"+output_file_name)
        
        if save_grammar:
            output_file_raport = "markov_processed_melody_grammar.txt"
            f = open(output_directory+"/"+output_file_raport, 'w')
            f.write(rules_raport)
            f.close()
            save_grammar = False

# ------------------------------------------------------------------------------------------------------
#                      Nastawy pracy algorytmu (TUTAJ EDYTOWAC)
# ------------------------------------------------------------------------------------------------------

print("Zadanie 2: gramatyki stochastyczne")

# nazwy folderów z których czytane są przykóady wykorzystywane do generowania modelu oraz do których
# zapisywane są wynikowe melodie wytworzone przez gramatykę
input_folder    = "music_examples"
output_folder   = "output_examples"

# rząd modelu - odpowiada za to ile poprzednich nut wpływa na rozkład pradopodobieństwa pojawienia się następnej
# nuty, dla wartości 1 mamy przypadek łańcucha Markowa
model_order     = 4

# wygenerowanie melodii na bazie przykładów z folderu ze zbiorem uczącym
# możliwe jest ustawienie długości generowanych nut, 1 - ćwierćnuta, 0.5 ósemka, 4 - cała nuta itd.
process_melodies(input_folder, output_folder, model_order, note_duration=1)