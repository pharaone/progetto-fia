from data_cleaning import DataCleaning
from random_forest_model import RandomForestModel

def main():
    # Percorso al CSV già pulito (adatta in base alla tua struttura di progetto)
    csv_path = "data/spaceship_titanic_clean.csv"

    # 1) Caricamento dei dati puliti
    cleaner = DataCleaning(csv_path)
    X, y = cleaner.get_features_and_target()  # X = features, y = target (Transported)

    # 2) Inizializza RandomForestModel con i parametri desiderati
    rf = RandomForestModel(
        n_splits=5,              # Numero di fold per la cross-validation
        random_state=42,         # Random seed per riproducibilità
        n_estimators=300,        # Numero di alberi nella foresta
        max_depth=None,          # Profondità massima (None = fino a foglie pure)
        n_jobs=-1,               # Usa tutti i core disponibili
        save_csv=False,          # True se vuoi salvare le metriche per fold in un CSV
        csv_path="metrics_per_fold.csv"
    )

    # 3) Addestra e valuta con Stratified K-Fold
    out = rf.train_and_evaluate(X, y)

    # 4) Stampa le metriche medie sui fold
    print("\n=== RIEPILOGO MEDIA K-FOLD ===")
    mean_metrics = out["mean_metrics"]
    for k, v in mean_metrics.items():
        if k == "Confusion Matrix (sum)":
            print(f"{k}:\n{v}")  # Stampa matrice di confusione aggregata
        else:
            print(f"{k}: {v:.4f}")  # Stampa metriche scalari con 4 decimali

if __name__ == "__main__":
    main()

