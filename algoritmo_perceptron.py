import random
import matplotlib.pyplot as plt

dataset_caratteristiche = [
[1, 0],
[0, 2],
[1, 1],
[2, 1],
[1, 3],
[2, 2],
[3, 2],
[2, 3]
]

ETICHETTE = ["felice", "triste"]
# 0 = triste, 1 = felice
dataset_etichette = [0, 0, 0, 0, 1, 1, 1, 1]

n_epoche = 50

tasso_apprendimento = 0.2

def disegna_dataset():

    felice = False
    triste = False
    for i in range(len(dataset_caratteristiche)):
        if dataset_etichette[i] == 1:
            plt.scatter(dataset_caratteristiche[i][0], dataset_caratteristiche[i][1], s = 100,
                   color = 'yellow', label=ETICHETTE[0] if felice == False else "",
                   edgecolor = 'k',
                   marker = '^')
            felice = True

        else:
            plt.scatter(dataset_caratteristiche[i][0], dataset_caratteristiche[i][1], s = 100,
                   color = 'red', label=ETICHETTE[1] if triste == False else "",
                   edgecolor = 'k',
                   marker = 'o')
            triste = True
            
    plt.xlabel('crack')
    plt.ylabel('doink')
    plt.legend()

def disegna_retta(pesi, bias, color="black"):

    x_1 = []
    x_2 = []
    
    for i in range(len(dataset_caratteristiche)):
        if pesi[0] != 0.0:
            x_1.append((-bias - pesi[1]*dataset_caratteristiche[i][1])/pesi[0])
        else:
            x_1.append(-bias - pesi[1]*dataset_caratteristiche[i][1]) 
        if pesi[1] != 0.0:
            x_2.append((-bias - pesi[0]*float(x_1[i]))/pesi[1])
        else:
            x_2.append(-bias - pesi[0]*float(x_1[i]))
            
    plt.plot(x_1, x_2, color)
    
# CLASSIFICATORE PERCEPTRON
def classificatore():
    
    # Pesi
    pesi = []
    """a = 1
    b = 1
    c = -3.5
    a = 4
    b = 4
    c = -4.0
    """
    a = float(random.choice([1, 2, 3, 4]))
    b = float(random.choice([1, 2, 3, 4]))
    # Bias
    c = float(random.choice([-4, -3, -2, -1, 1]))
    pesi.append(a)
    pesi.append(b)
    
    print("Pesi", pesi, "bias", c)


    n = 0
    while n < n_epoche:
        posizioni_punti_errati, y_prev = previsione(pesi, c)
        
        if not posizioni_punti_errati:
            break
        # Scelgo un punto random dalla lista dei punti errati per ricalcolare pesi e bias
        punto_random = random.choice(posizioni_punti_errati)
        
        pesi, c = apprendimento(pesi, c, punto_random, y_prev)
        disegna_retta(pesi, c)

        n += 1
    
    print("Pesi", pesi, "bias", c)
    return pesi, c

def previsione(pesi, bias):
    
    punteggi = []
    y_prev = []
    
    for i in range(len(dataset_caratteristiche)):
        punteggi.append(pesi[0]*dataset_caratteristiche[i][0] + pesi[1]*dataset_caratteristiche[i][1] + bias)
        
        print("Punteggio:", punteggi[i])
        y_prev.append(attivazione(punteggi[i]))
        print("Previsione:", y_prev[i])
        
    posizioni_punti_errati = calcolo_errore(punteggi, y_prev)
    
    return posizioni_punti_errati, y_prev

def apprendimento(pesi, bias, punto_random, y_prev):
    
    pesi[0] = pesi[0] + tasso_apprendimento*(dataset_etichette[punto_random]-y_prev[punto_random]*dataset_caratteristiche[punto_random][0])
    pesi[1] = pesi[1] + tasso_apprendimento*(dataset_etichette[punto_random]-y_prev[punto_random]*dataset_caratteristiche[punto_random][1])
    bias = bias + tasso_apprendimento*(dataset_etichette[punto_random]-y_prev[punto_random])

    return pesi, bias

def attivazione(punteggi):
    if punteggi >= 0:
        return 1
    else:
        return 0

def calcolo_errore(punteggi, y_prev):
    
    errori = []
    posizioni_punti_errati = []
    for i in range(len(dataset_etichette)):
        
        if y_prev[i] != dataset_etichette[i]:
            errori.append(abs(punteggi[i]))
            posizioni_punti_errati.append(i)
            
        print("Previsto", y_prev[i], "con etichetta", dataset_etichette[i])
        
    print("Errori", errori)
    print("Errore medio:", errore_medio(errori))

    return posizioni_punti_errati

def errore_medio(x):
    m = s = 0.0
    if not x:
        return 0
    
    n = len(x)

    for ele in x:
        s += ele
    m = s/n
    
    return m

def main():

    disegna_dataset()

    pesi, bias = classificatore()
    
    disegna_retta(pesi, bias, "red")
    plt.show()

main()