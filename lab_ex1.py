#-*- coding:utf8 -*-

# ------------------------------------------------------------------------------------------------------
#   Sekcja importu bibliotek:
#
#   Fragment do modyfikacji w ramach ćwiczenia znajduje się na dole, na koncu niniejszego pliku.

import sys
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import scipy.io.wavfile as wav
import scipy.signal as signal

sys.path.append('../_utils')
from MIDI_IO import *

# ------------------------------------------------------------------------------------------------------
#   Algorytmy generowania szumów
#  ------------------------------------------------------------------------------------------------------

# ---------------------------------
#   Algorytm generowania szumu białego
#       jako parametr wejściowy podawana jest oczekiwana długość ciągu próbek szumu
#
def generate_white_noise(number_of_samples):
    return np.random.uniform(-1, 1, number_of_samples)

# ---------------------------------
#   Algorytm generowania szumu Browna
#       jako parametr wejściowy podawana jest oczekiwana długość ciągu próbek szumu
#
def generate_brownian_noise(number_of_samples):
    # punktem wyjścia do generowania szumu Browna jest zwykły szum biały o
    # płaskim rozkładzie prawodopodobieństw generowanych wartości, tj.
    # każda wartość z przedziału -1 do 1 jest tak samo prawdopodobna
    white_noise_input       = np.random.uniform(-1, 1, number_of_samples)

    # szum Browna otrzymuje się poprzez dodawanie do siebie kolejnych próbek
    # szumu białego, jest to tzw. akumulata i odpowiada ona całkowaniu sygnału
    # w dziedzinie sygałów ciągłych
    #
    # pierwszym krokiem w tym procesie jes przygotowanie pustej listy i dodanie tam
    # pierwszej próbki sygnału szumy białego
    brownian_noise_ouput    = []
    brownian_noise_ouput.append(white_noise_input[0])

    # następnie w pętli do listy dołączamy kolejne próbki szumu Browna, które są
    # sumą poprzednio wygenerowanej próbki Browna oraz kolejnych próbek szumu białego
    for i in range(1,len(white_noise_input)):
        brownian_noise_ouput.append(brownian_noise_ouput[-1] + white_noise_input[i])

    return brownian_noise_ouput

# ---------------------------------
#   Algorytm generowania szumu różowego
#       jako parametr wejściowy podawana jest oczekiwana długość ciągu próbek szumu
#       oraz liczba generatorów składowych, odpowiadają one za dokładność z
#       jaką generowany jest szum różowy (algorytm Vossa-McCartneya)
#
def generate_pink_noise(number_of_samples, gen_levels=10):

    # szum różowy otrzymuje się jako sumę składową sygnałów generowanych przez generatory szumu białego
    # każdy kolejny generator ma dwa razy dłuższy okres niż poprzedni oraz startuje w połowie swojego oresu
    # wizualnie zmiany kolejnych wartości wytawrzanych przez kolejne generatory można przedstawić następująco:
    #
    # xxxxxxxxxxxxxxx
    # x x x x x x x x
    #  x   x   x   x
    #    x       x
    #        x

    # x oznacza zmianę wartości na wyjściu generatora

    # więcej informacji dotyczących generatora Vossa_McCartneya znajduje się na stronie:
    # http: // www.firstpr.com.au / dsp / pink - noise /

    # obiekt generatora, który wytwarza szum biały, którego próbki zmieniają się co period kroków
    # możliwe jest także wybranie początkowej fazy, tak aby móc je ustawić zgodnie z wymogami algorytmu Vosa-McCartneya
    class level_generator:
        def __init__(self, period, phase=0):
            self.period             = period    # okres generatora
            self.phase              = phase     # faza początkowa generatora - jako liczba od 0 do period-1
            self.current_sample     = np.random.uniform(-1, 1, 1)[0] # startowa wartośc próbki

        def get_sample(self):
            # jeżeli wartość fazy jest większa niż okres generatora, wygeneruj nową wartość na wyjściu generatora
            # oraz wyzeruj fazę
            if self.phase >= self.period:
                self.current_sample = np.random.uniform(-1, 1, 1)[0]
                self.phase          = 0

            # zwiększ wartość fazy o 1, do czasu gdy nie zostanie przekroczona wartość period-1 próbka na wyjściu
            # generatora (current_sample) nie zostanie zmieniona.
            self.phase +=1
            return self.current_sample

    # precyzja formatu double pozwala na wykorzystanie maksymalnie 30 generatorów, dla większej ich liczby
    # maksymalna wielkość możliwa do zapisania w zmiennej period obiektu level_generator jest zbyt duża.
    if gen_levels > 30:
        gen_levels = 30

    # wytworzenie generatorów szumu białego
    generators_table = []
    for i in range(0,gen_levels):
        # każdy kolejny generator ma dwa razy większy okres niż poprzedni i startuje w połowie swojego okresu
        period = np.power(2, i)
        generators_table.append(level_generator(period, np.floor(period/2.0)))

    # przygotuj zmienną na wynikowe wartości szumu różowego
    pink_noise_output = []
    for i in range(0,number_of_samples):
        # początkowo wartość próbki ustawiamy na zero, będziemy do tej wartości dodawać liczby
        # wytworzone przez kolejne generatory
        pink_noise_sample = 0

        # pobierz wartość z każdego generatora i dodaj ją do pink_noise_sample
        for generator in generators_table:
            pink_noise_sample += generator.get_sample()

        # tak wygenerowaną wartość dodaj do przygotowanego wcześniej wektora na wynikowe próbki szumu
        # aby ograniczyć wartości szumu wynikowego ostateczny wynik dzielimy przez liczbę generatorów
        pink_noise_output.append(pink_noise_sample/float(gen_levels))
    return pink_noise_output

# ------------------------------------------------------------------------------------------------------
#                      Pomocnicze funkcje do przetwarzania danych z generatorów szumu
# ------------------------------------------------------------------------------------------------------

# ---------------------------------
# funkcja przetwarza wartości z generatorów (posiadają one zakres wartości od -1 do 1, ciągły) na dyskretny
# zbiór wartości od zera do span, który można przesunąć o wartość podaną w parametrze shift
def quantize_noise_sample(noise_samples, span, shift):
    # przygotuj pustą listę na zwracane wartości
    output_values       = []

    # przesuń wartości szumu tak, by minimalną wartością było zero
    normalized_noise    = noise_samples - np.min(noise_samples)
    # znormalizuj szum tak, by maksymalną wartością było 1
    normalized_noise    = normalized_noise/np.max(normalized_noise)
    # przemnóż przez wartość zmiennej span, tak aby maksymalna wartość była równa span
    normalized_noise    = normalized_noise*span

    # do wcześniej przygotowanej listy załaduj próbki z poprzednich kroków przesunięte
    # o wielkość podaną w zmiennej shift
    for sample in normalized_noise:
        output_values.append(np.floor(sample)+shift)

    return output_values

# ---------------------------------
#   przetworzenie szumu z generatora na melodię możliwą do zapisu jako plik MIDI
#
#   noise_samples - próbki szumu wygenerowane przy pomocy jednego z wcześniej opisanych generatoró
#
#   note_span - krotka mówiąca od jakiej wartości do jakiej wartości mają rociągać się generowane
#   wartości szumu, np. zapis (20,80) mówi, że generowane maja być nuty z wysokościami o kodach MIDI
#   od 20 do 80
#
#   note_duration - długośći nut generowanych przez generator, 1 oznacza ćwierćnutę, 4 całą nutę, 0.5 ósemkę itd.
def noise_to_melody(noise_samples, note_span, note_duration=1):

    # przelicz parametr podany w note_span na wielkości akceptowane przez funkcję
    # quantize_noise_sample
    span                = note_span[1] - note_span[0]
    shift               = note_span[0]

    # oblicz dyskretne wartości odpowiadające wysokościom kolejnych generowanych nut
    normalized_noise     =  quantize_noise_sample(noise_samples, span, shift)

    # na bazie poprzednio wygenerowanych wysokości nut utwórz listę nut
    # każda nuta ma jednakową długość podaną w parametrze note_duration
    output_data = []
    for sample in normalized_noise:
        output_data.append(Note(sample,note_duration))

    return output_data


# ---------------------------------
#   obliczenie funkcji autokorelacji sygnału
#   zrealizowane jako szybko splot dwóch kopii sygnału (w tym jednej obróconej na osib czasu
#   za pomocą funkcji flipud)
def autocorrelation_of_noise(noise_samples):
    plt.figure()
    plt.plot(signal.fftconvolve(np.flipud(noise_samples), noise_samples))
    plt.xlabel("przesuniecie [probka]")
    plt.ylabel("f-cja autokorelacji [-]")
    plt.title("funkcja autokorelacji")
    plt.grid()

# ---------------------------------
#   obliczenie widma amplitudowego sygnału
def calculate_spectrum(noise_samples):
    plt.figure()

    frequency_response = np.abs(np.fft.fft(noise_samples))
    frequency_labels = np.arange(0,len(frequency_response)/2.)/float(len(frequency_response))

    plot_length = int(np.floor(len(frequency_response)/2))
    plt.plot(frequency_labels, 20*np.log10(frequency_response[0:plot_length]))
    plt.xlabel("czestotliwosc znormalizowana [-]")
    plt.ylabel("amplituda [dB]")
    plt.title("widmo szumu")
    plt.grid()


# ------------------------------------------------------------------------------------------------------
#                      Nastawy pracy algorytmu (TUTAJ EDYTOWAC)
# ------------------------------------------------------------------------------------------------------

print("Zadanie 1: generatory szumu")

# ile nut ma zawierać plik midi wygenerowany przez skrypt?
length_of_records = 100

# noise_samples zawiera próbki szumu wykorzystane później do wygenerowania nut
# aby zmienić rodzaj generatora starczy zakomentować lub odkomentować jedną z linijek poniżej:
# w przypadku szumu różowego można regulować liczbe generatorów wykorzystywanych przez algorytm generatora
#
noise_samples = generate_white_noise(length_of_records)
# noise_samples = generate_brownian_noise(length_of_records)
# noise_samples = generate_pink_noise(length_of_records,10)

# zakres genrowanych wysokości nut: (numer_midi_nuty_najnizszej, numer_midi_nuty_najwyzszej)
note_span = (50,80)

# długość generowanych nut: (1 - ćwierćnuta, 0.5 ósemka, 4 - cała nuta itd.)
note_length = 0.5

# nazwa pliku z wygenerowanym ciągiem nut
output_file_name = "noise_based_generator_output.mid"

# zapis gotowej sekwencji nut do pliku MIDI
save_melody_to_midi(noise_to_melody(noise_samples,note_span, note_length), output_file_name)

