# progetto‑fia

Repository del progetto per il corso Fondamenti di Intelligenza Artificiale (FIA)
Autori: Emanuele Antonio Faraone, dariopattu, Stefano Imbalzano 
GitHub

## Descrizione 

Questo progetto ha lo scopo di sperimentare metodi di classificazione e clustering su dati testuali (descrizioni, autori, ecc.), con l’obiettivo di valutare quale approccio produce risultati migliori nella categorizzazione o raggruppamento dei documenti.

Il programma principale (main.py) gestisce automaticamente la pipeline completa:

Analisi iniziale del dataset

Pre-elaborazione del testo

Addestramento di vari modelli di classificazione e clustering

Valutazione dei modelli tramite metriche e visualizzazioni

Sono inclusi anche diagrammi di confusione per i modelli (e.g. confusion_adaboost.png, confusion_rf.png) e la matrice di correlazione (correlation_matrix.png) per illustrare le relazioni tra variabili. 
GitHub

Struttura del repository

.
├── adaboost/                   # Modelli o codice legato ad AdaBoost  
├── preprocessing/              # Script per pulizia e trasformazione del testo  
├── randomforest/               # Modelli o codice legato a Random Forest  
├── confusion_adaboost.png      # Plot risultato confusione AdaBoost  
├── confusion_rf.png            # Plot risultato confusione Random Forest  
├── correlation_matrix.png      # Plot matrice di correlazione  
├── main.py                      # Script principale che esegue il flusso completo  
├── requirements.txt             # Dipendenze del progetto  
└── .gitignore                   # File ignorati dal repository  

Requisiti

Prima di eseguire il progetto, assicurati di avere:

Python (versione compatibile, ad esempio 3.x)

Le librerie Python elencate in requirements.txt (esempi possibili: pandas, numpy, scikit-learn, matplotlib, nltk, ecc.) 
GitHub

Puoi installare le dipendenze con:
pip install -r requirements.txt

Istruzioni d’uso

1.Clona il repository:

2.Installa le dipendenze (vedi sezione “Requisiti”).

3.Esegui lo script principale:
python main.py

Lo script:

visualizza statistiche e informazioni di base sul dataset

effettua il preprocessing

addestra vari modelli

produce metriche di valutazione e plot

Risultati attesi

Al termine dell’esecuzione, ti aspettano:

Output testuali con metriche (accuratezza, precisione, recall, F1, ecc.)

Grafici salvati (matrici di confusione, correlazioni)

Confronto tra i modelli per verificare quale approccio risulta più efficace per il dataset adottato