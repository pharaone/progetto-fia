# progetto‑fia

Repository del progetto per il corso Fondamenti di Intelligenza Artificiale (FIA)
Autori: _Emanuele Antonio Faraone, Dario Pattumelli, Stefano Imbalzano_

## Descrizione 

Questo progetto ha lo scopo di sperimentare metodi di intelligenza artificiale per predirre se un certo passeggero è stato teletrasportato da un anomalia.
I dati e la descrizione della _challenge_ si trovano su www.kaggle.com/competitions/spaceship-titanic

Il programma principale (_main.py_) gestisce automaticamente la pipeline completa:

* Lettura del file csv in input
* Preprocessing dei dati
* Hyper parameter tuning per **RandomForest** e **AdaBoost**
* Fit dei modelli con K-fold validation
* Collezione metriche, grafici e risultati

## Struttura del repository

.
├── adaboost/                   **# Codice legato ad AdaBoost**  
├── preprocessing/              **# Script per il preprocessing del dataset**  
├── randomforest/               **# Codice legato a Random Forest**  
├── confusion_adaboost.png      **# Plot risultato confusione AdaBoost**  
├── confusion_rf.png            **# Plot risultato confusione Random Forest**  
├── correlation_matrix.png      **# Plot matrice di correlazion**e  
├── main.py                      **# Script principale che esegue il flusso completo**  
├── requirements.txt             **# Dipendenze del progetto**

## Requisiti e esecuzione del progetto

Prima di eseguire il progetto, assicurati di avere Python in una versione compatibile, ad esempio 3.12.x)

Clona il repository o scarica il progetto

Puoi installare le dipendenze con:
`pip install -r requirements.txt`

Dopo di ciò assicurati di aver copiato nella main folder del progetto il file train.csv fornito da Kaggle ed averlo rinominato input.csv

Arrivato qui puoi eseguire lo script `python main.py`

## Risultati

Output testuali con metriche (accuratezza, precisione, sensitivity) nel file di output _metrics_summary.csv_

Matrici di confusione per entrambi i modelli adottati (AdaBoost e RandomForest), matrice di correlazione tra le feature.

Confronto tra i modelli per verificare quale approccio risulta più efficace per il dataset adottato