import pandas as pd
import numpy as np

def predice_naivebayes(email, emails, model, etichetta):
    
    # numero totale di email
    totale = len(emails)
    # numero totale di email di spam
    num_spam = sum(emails['spam'])
    # numero totale di email di ham
    num_ham = totale - num_spam
    
    # trasformiamo le email in una lista di parole minuscole
    email = email.lower()
    words = set(email.split())
    spams = [1.0]
    hams = [1.0]
    
    print("Numero totale di email di spam:", num_spam)
    print("Numero totale di email di ham:", num_ham)

    for word in words:
        if word in model:
            # per ogni parola
            # calcola la probabilità a posteriori che un messaggio di spam o di ham contenenga quella parola (vedere slide 8)
            # si divide le occorrenze della parola nelle email di spam per il numero totale di email di spam
            spams.append(model[word]['spam']/num_spam*totale) # si moltiplica per il totale per non avere prodotti di probabilità troppo piccoli
            # P(x1, x2,..., xn | spam) P(x1, x2,..., xn | ham)
            # si divide le occorrenze della parola nelle email di ham per il numero totale di email di ham
            hams.append(model[word]['ham']/num_ham*totale)
    
    # moltiplica tutte le probabilità precedenti con la probabilità a priori che l'email sia di spam o di ham
    # la probabilità a priori è il numero di email di spam o di ham (vedere slide 14)
    prod_spams = np.long(np.prod(spams)*num_spam) # P(x1, x2,..., xn | spam) * P(spam)
    prod_hams = np.long(np.prod(hams)*num_ham) # P(x1, x2,..., xn | ham) * P(ham)
    
    # normalizza le due probabilità per renderle a somma 1 (usando il teorema di Bayes) e restituisce il risultato
    # che sarebbe la probabilità condizionata che date determinate parole la email sia di spam o no, vedi slide 16
    if etichetta == "spam":
        previsione = prod_spams/(prod_spams + prod_hams) # P(spam | x1, x2,..., xn)
    else:
        previsione = prod_hams/(prod_spams + prod_hams)
        
    return previsione

# Tutto minuscolo e dividiamo i testi in elenchi di parole
def process_email(testo):
    testo = testo.lower()
    # Trasformiamo prima in set() per eliminare i duplicati perchè ci interessa solo che una parola compaia almeno una volta nella e-mail
    return list(set(testo.split()))


# Calcola la probabilità a priori che una email sia di spam
def calc_apriori_spam(emails):
    
    # Calcola il numero di email di spam diviso per il numero totale di email
    return sum(emails['spam'])/len(emails)

# Calcola la probabilità a posteriori che una email sia di spam o di ham data una specifica parola 
def cal_aposteriori(tipo, occ_parola):
    
    p_aposteriori = occ_parola[tipo]/ (occ_parola['spam'] + occ_parola['ham'])
    
    return p_aposteriori
    
def calc_occorrenze_parole(emails):
    
    model = {}
    
    # Registriamo le occorrenze di tutte le parone in un dizionario dividendo tra email di spam e ham
    for index, email in emails.iterrows():
        for word in email['words']:
            if word not in model:
                # iniziamo da 1 per evitare di dividere per 0
                model[word] = {'spam': 1, 'ham': 1}
            if word in model:
                if email['spam']:
                    model[word]['spam'] += 1
                else:
                    model[word]['ham'] += 1
                    
    return model
    
def main():
    emails = pd.read_csv("emails.csv")
    print(emails[0:5])

    # applichiamo le modifiche ad una nuova colonna
    emails['words'] = emails['text'].apply(process_email)
    print(emails[0:5])
    
    # calcola a priori spam
    p_spam = calc_apriori_spam(emails)
    print("Probabilità a priori che un'email sia di SPAM:",p_spam)
    
    # calcola probabilità complementare: che una email sia di ham
    p_ham = 1 - p_spam
    print("Probabilità a priori che un'email sia di HAM:",p_ham)
    
    # calcola occorrenze delle parole nelle email di spam e ham
    word_test = "sale"
    dizionario_occorrenze = calc_occorrenze_parole(emails)
    print("Occorrenze della parola", word_test,":", dizionario_occorrenze[word_test])
    
    # calcola probabilità a posteriori che data la parola 'sale' l'email sia di spam
    p_aposteriori = cal_aposteriori('spam', dizionario_occorrenze[word_test])
    print("Probabilità che la parola", word_test,"sia di SPAM:", p_aposteriori)
    
    #
    email_test = "buy cheap money lottery sale satisfaction"
    prob_cond = predice_naivebayes(email_test, emails, dizionario_occorrenze, "spam")
    print("Email di prova:", email_test)
    print("Probabilità che l'email sia di SPAM:",prob_cond)
    
main()


