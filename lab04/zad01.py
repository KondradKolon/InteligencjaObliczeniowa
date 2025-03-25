import math
Bias_1_a = 0.80109
Bias_1_b = 0.43529
Bias_2 = -0.2368


def activationFunction(number):
    return 1/(1+math.e**(number*-1))


def forwardPass(wiek, waga, wzrost):
    neuron_1 = (-(wiek * 0.46122)) + (waga*0.97314) + \
        (-1*(wzrost*0.39203)) + Bias_1_a
    neuron_1_act = activationFunction(neuron_1)
    neuron_2 = (wiek*0.78548) + (waga * 2.10584) + (wzrost*-0.57847) + Bias_1_b
    neuron_2_act = activationFunction(neuron_2)
    wynik = (neuron_1_act * -0.81546) + (neuron_2_act*1.03775) + Bias_2

    return wynik


tabela = [[23, 75, 176], [25, 67, 180], [
    28, 120, 175], [22, 65, 165], [46, 70, 187]]

for osoba in tabela:
    print(forwardPass(osoba[0], osoba[1], osoba[2]))


