Space Ship Titanic

COME FUNZIONA (IL PERCORSO LOGICO)
Il percorso inizia dal file input.csv, che contiene le osservazioni grezze. Il dataset viene trasformato: si codificano i booleani, si estraggono informazioni da PassengerId e Cabin, si sintetizzano le spese, si imputano i mancanti, si codificano le categorie e si rimuovono ridondanze tramite una soglia di correlazione. A questo punto i due modelli vengono prima sintonizzati (GridSearchCV) e poi addestrati con una Stratified K-Fold. Le predizioni out-of-fold consentono di calcolare metriche oneste e riutilizzabili (accuracy media e incertezza, precisione, specificità), oltre alle confusion matrix. Infine, tutto viene salvato: dataset preprocessato, immagini delle confusion matrix e tabella delle metriche.

DATI ATTESI E OBIETTIVO
Il progetto assume la presenza di variabili tipiche del problema Spaceship Titanic: Transported (target), identificativi come PassengerId e Name, Cabin in formato “Deck/Numero/Side”, categorie (HomePlanet, Destination) e spese (RoomService, FoodCourt, ShoppingMall, Spa, VRDeck). L’obiettivo è prevedere Transported in modo affidabile, mantenendo il controllo sulla varianza dei risultati attraverso la validazione incrociata.

PREPROCESSING

Codifica booleane: Transported, VIP e CryoSleep diventano 0/1.

Feature di gruppo: da PassengerId si ricava un identificatore di gruppo e la dimensione del gruppo, utile a catturare relazioni tra passeggeri.

Feature da Cabin: si estraggono parti di Cabin e si deriva Cabin_region (bin di posizione), tenendo un approccio robusto agli “Unknown”.

Spese log-trasformate: le spese vengono riempite a 0, trasformate con log1p e sommate in una variabile aggregata Expenditures, per mitigare code pesanti e mettere le cifre su scala più gestibile.

Imputazione: KNNImputer stima i mancanti sulle sole colonne numeriche, sostituendoli con valori coerenti col vicino più simile.

Categorical encoding: OneHotEncoder con drop della prima categoria e gestione esplicita degli “Unknown”.

Riduzione ridondanze: una heatmap guida la rimozione automatica delle feature troppo correlate (soglia predefinita 0.8), riducendo il rischio di sovrapposizioni informative.

MODELLI E FILOSOFIA DI APPRENDIMENTO
RandomForest apprende una collezione di alberi decisionali in parallelo e media le loro predizioni. Il parametro max_depth=None non significa “alberi di profondità nulla”, ma “nessun limite di profondità”: gli alberi possono crescere finché le regole di stop lo consentono. Questo, insieme a n_estimators e max_features, regola il compromesso tra bias e varianza, oltre alla robustezza a feature rumorose.

AdaBoost combina molti “weak learners” (decision tree poco profondi) in un classificatore più forte, ri-ponderando progressivamente gli esempi difficili. I parametri chiave sono n_estimators (quanti deboli combinare), learning_rate (quanto variare i pesi a ogni iterazione) e la profondità massima del base learner (base_max_depth), tipicamente bassa per mantenere la debolezza richiesta dall’algoritmo.

VALIDAZIONE, SCALING E PREVENZIONE DEL LEAKAGE
La valutazione usa StratifiedKFold per preservare la proporzione tra classi in ogni split. Per ogni fold, le sole colonne numeriche del training vengono standardizzate (StandardScaler) e i parametri del scaler sono riutilizzati sul validation del medesimo fold. In questo modo si evita leakage informativo tra train e validation e si mantiene un flusso realistico di generalizzazione.

SINTONIZZAZIONE DEGLI IPERPARAMETRI
Entrambi i modelli offrono un metodo di tuning tramite GridSearchCV.
Per RandomForest, la griglia include n_estimators, max_depth, max_features e class_weight (opzionale “balanced” se le classi sono sbilanciate).
Per AdaBoost, la griglia tocca n_estimators, learning_rate e la profondità del base learner.
Il punteggio predefinito è l’accuracy, ma può essere sostituito con metriche più adatte (ad esempio f1 o roc_auc) se il problema lo richiede.

METRICHE E LETTURA DEI RISULTATI
Il progetto riporta:
— Accuracy media sui fold, insieme alla deviazione standard per capire la variabilità da split a split.
— SEM (Standard Error of the Mean) dell’accuracy, cioè la deviazione standard divisa per la radice del numero di split: fornisce l’incertezza associata alla media stimata. “Accuracy ± SEM” esprime quanto ci fidiamo di quella media.
— Precision (quanto sono “pulite” le predizioni positive) e Specificità (quota di veri negativi correttamente riconosciuti).
— Confusion matrix out-of-fold complessiva, calcolata a una soglia default di 0.5: mostra a colpo d’occhio dove il modello sbaglia (falsi positivi/negativi).

COSA PRODUCE IL CODICE
Al termine dell’esecuzione vengono salvati:
— my_file.csv: il dataset preprocessato, pronto per analisi ulteriori o inferenza.
— confusion_rf.png e confusion_adaboost.png: le confusion matrix dei due modelli, utili per slide e report.
— metrics_summary.csv: una tabella riassuntiva con n_splits, threshold, accuracy_mean, accuracy_std, accuracy_sem, precision e specificity.
— correlation_matrix.png: la heatmap delle correlazioni tra feature numeriche, utile per documentare la riduzione di ridondanza.

COME SI ESEGUE

Posiziona input.csv nella root del progetto.

Esegui il main con Python. Non occorrono parametri a riga di comando, le impostazioni di default sono sensate per una prima corsa.

A fine esecuzione, leggi i log in console per conoscere i migliori iperparametri trovati e controlla i file prodotti nella cartella di lavoro.

CONFIGURABILITÀ E PERSONALIZZAZIONE
Le classi di configurazione (Config per RandomForest e ConfigAdaBoost per AdaBoost) espongono i parametri principali: numero di split, seme random, iperparametri del modello e colonne da escludere. Le griglie del GridSearch possono essere modificate a piacere per ampliare o restringere l’esplorazione. Se il dataset risulta sbilanciato, è possibile attivare class_weight="balanced" in RandomForest o cambiare la metrica di ottimizzazione in GridSearch per focalizzarsi su precision, recall o f1.

BUONE PRATICHE E SPUNTI DI ESTENSIONE
— Valuta metriche aggiuntive (recall, f1, ROC-AUC) e, se necessario, uno sweep sulla soglia di decisione per massimizzare la metrica che ti interessa.
— Considera RandomizedSearchCV o ottimizzazioni bayesiane se la griglia diventa ampia.
— Salva i modelli per fold e crea un semplice ensemble in inferenza (già supportato in AdaBoostCV con la media delle proba dei fold).
— Se il problema è fortemente sbilanciato, esplora tecniche di resampling (SMOTE, undersampling) o metriche di ottimizzazione alternative.
— Documenta la versione dei pacchetti: la riproducibilità è già facilitata da random_state e dallo schema di validazione, ma le versioni dei pacchetti restano determinanti.

MESSAGGIO CHIAVE
Il progetto punta alla qualità della valutazione più che al singolo punteggio: usare predizioni out-of-fold, riportare media, deviazione standard e SEM rende i risultati più trasparenti e confrontabili. La pipeline di preprocessing è pensata per essere pragmatica (feature utili, imputazione robusta, encoding pulito) e i due modelli offrono un buon compromesso tra performance e interpretabilità operativa. Se ti servono risultati pronti per una presentazione, i file generati (CSV e immagini) sono già formattati per essere inseriti in report e slide.
