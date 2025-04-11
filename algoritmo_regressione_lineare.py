from matplotlib import pyplot as plt
import numpy as np
import random

# Imposto il seed per ottenere sempre lo stesso numero casuale
random.seed(0)

def disegna_linea(slope, y_intercept, color='grey', linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, y_intercept + slope*x, linestyle='-', color=color, linewidth=linewidth)

def disegna_punti(features, labels, x_label = "numero di locali", y_label = "prezzi"):
    X = np.array(features)
    y = np.array(labels)
    plt.scatter(X, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
# Usiamo la regressione lineare per avvicinare una linea a un punto
# utilizzando traslazioni e rotazioni
def trucco_semplice(prezzo_base, prezzo_per_stanza, num_stanze, prezzo):
    
    num_casuale_piccolo_1 = random.random()*0.1
    num_casuale_piccolo_2 = random.random()*0.1
    
    # Previsione p^ = mr + b (equzione della retta y = mx + b)
    previsione_prezzo = prezzo_per_stanza*num_stanze + prezzo_base
    
    # Se il punto è sopra la retta e a destra dell'asse y
    if prezzo > previsione_prezzo and num_stanze > 0:
        # Ruota in senso antiorario e trasla in verso l'alto
        prezzo_per_stanza += num_casuale_piccolo_1
        prezzo_base += num_casuale_piccolo_2
        
    # Se il punto è sopra la retta e a sinistra dell'asse y        
    if prezzo > previsione_prezzo and num_stanze < 0:
        # Ruota in senso orario e trasla verso l'alto
        prezzo_per_stanza -= num_casuale_piccolo_1
        prezzo_base += num_casuale_piccolo_2
        
    # Se il punto è sotto la retta e a destra dell'asse y        
    if prezzo < previsione_prezzo and num_stanze > 0:
        # Ruota in senso orario e trasla verso il basso
        prezzo_per_stanza -= num_casuale_piccolo_1
        prezzo_base -= num_casuale_piccolo_2
        
    # Se il punto è sotto la retta e a sinistra dell'asse y        
    if prezzo < previsione_prezzo and num_stanze < 0:
        # Ruota in senso antiorario e trasla verso il basso
        prezzo_per_stanza += num_casuale_piccolo_1
        prezzo_base -= num_casuale_piccolo_2
        
    return prezzo_per_stanza, prezzo_base

def trucco_assoluto(prezzo_base, prezzo_per_stanza, num_stanze, prezzo, learning_rate):
    
    # Previsione p^ = mr + b (equzione della retta y = mx + b)
    previsione_prezzo = prezzo_per_stanza*num_stanze + prezzo_base
    
    # Se il punto è sopra la retta
    if prezzo > previsione_prezzo:
        # Aggiungo tasso di apprendimento * r alla pendenza, cioè ruoto la retta in senso antioriario se il punto è a destra di y
        # o in senso orario se il punto è a sinistra dell'asse y
        prezzo_per_stanza += learning_rate*num_stanze
        # Aggiungo il tasso di apprendimento all'intercetta, cioè traslo la retta verso l'alto
        prezzo_base += learning_rate
    # Se il punto è sotto la retta
    else:
        # Sottraggo alla pendenza il tasso di apprendimento * r, cioè ruoto la retta in senso orario se il punto è a destra di y
        # o in senso antiorario se il punto è a sinistra dell'asse y
        prezzo_per_stanza -= learning_rate*num_stanze
        # Traslo la retta verso il basso
        prezzo_base -= learning_rate
        
    return prezzo_per_stanza, prezzo_base

def trucco_quadrato(prezzo_base, prezzo_per_stanza, num_stanze, prezzo, learning_rate):
    
    # Previsione p^ = mr + b (equzione della retta y = mx + b)
    previsione_prezzo = prezzo_per_stanza*num_stanze + prezzo_base
    
    # Utilizza un tasso di apprendimento per cambiare di piccole quantità durante l'addestramento
    
    # ruota
    # il valore r(p - p^) è positivo quando sia r (numero locali) che p - p^ sono entrambi positivi o negativi (ruotiamo in senso antiorario)
    prezzo_per_stanza += learning_rate*num_stanze*(prezzo-previsione_prezzo) # aggiorna la pendenza
    
    # trasla
    prezzo_base += learning_rate*(prezzo-previsione_prezzo)# se il punto è sopra la linea questa differenza
                                                           # è positiva, altrimenti è negativa
    
    return prezzo_per_stanza, prezzo_base


def regressione_lineare(features, labels, learning_rate=0.01, epochs = 1000):
    
    # Pendenza/m di partenza
    prezzo_per_locale = random.random()
    
    # Intercetta-y - b di partenza
    prezzo_base = random.random()
    
    print(prezzo_per_locale, prezzo_base)
    
    errors = []
        
    # Ripetiamo l'algoritmo per n epoche
    for epoch in range(epochs):
        # Commentare/decommentare per stampare diverse epoche della linea
        if epoch == 1:
        #if epoch <= 10:
        #if epoch <= 50:
        #if epoch > 50:
        #if True:
            disegna_linea(prezzo_per_locale, prezzo_base, starting=0, ending=8)
            
        # Salva le previsioni e gli errori
        previsioni = features[0]*prezzo_per_locale+prezzo_base
        
        errors.append(rmse(labels, previsioni))
        
        # Utilizza un elemento del dataset casuale
        i = random.randint(0, len(features)-1)
        num_stanze = features[i]
        prezzo = labels[i]
        
        # Commenta/decommenta per usare diversi algoritmi

        # prezzo_per_locale, prezzo_base = trucco_semplice(prezzo_base, prezzo_per_locale, num_stanze, prezzo)
        """prezzo_per_locale, prezzo_base = trucco_assoluto(prezzo_base, prezzo_per_locale, num_stanze, prezzo,
                                                  learning_rate=learning_rate)"""
        prezzo_per_locale, prezzo_base = trucco_quadrato(prezzo_base, prezzo_per_locale, num_stanze, prezzo,
                                                  learning_rate=learning_rate)
    print('Prezzo per stanza:', prezzo_per_locale)
    print('Prezzo base:', prezzo_base)
    disegna_linea(prezzo_per_locale, prezzo_base, 'black', starting=0, ending=8)
    disegna_punti(features, labels)

    plt.show()
    disegna_punti(range(len(errors)), errors, "Numero di epoche", "Errore numerico")
    plt.show()
    
    return prezzo_per_locale, prezzo_base

# Radice dell'errore quadratico medio
def rmse(labels, previsioni):
    n = len(labels)
    differenze = np.subtract(labels, previsioni)
    # Usiamo la radice quadrata delle differenze tra il valore reale e previsto
    # dot() prodotto scalare di un vettore con se stesso = somma dei quadrati degli elementi
    return np.sqrt(1.0/n * (np.dot(differenze, differenze)))

def main():
    
    # caratteristiche/pesi - r
    # numero di locali
    features = np.array([1,2,3,5,6,7])

    # etichette
    # prezzi delle case
    labels = np.array([155, 197, 244, 356,407,448])

    # Iperparametri
    learning_rate = 0.01
    epochs = 1000
    
    print(features)
    print(labels)

    disegna_punti(features, labels)
    
    # Valore limite dell'asse y
    plt.ylim(0,500)

    # Costruisce la retta
    prezzo_per_locale, prezzo_base = regressione_lineare(features, labels, learning_rate, epochs)

    # previsione per 4 stanze
    previsione = prezzo_per_locale * 4 + prezzo_base
    print("Previsione del prezzo per 4 stanze", previsione)

    features = np.append(features, 4)
    labels = np.append(labels, previsione)
    
    disegna_linea(prezzo_per_locale, prezzo_base, 'black', starting=0, ending=8)

    disegna_punti(features, labels)

main()