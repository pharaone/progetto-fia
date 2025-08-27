from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

@dataclass
class FoldResult:
    """
    Struttura dati per memorizzare i risultati di un fold.
    """
    fold: int
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    error_rate: float
    precision: float
    recall: float            # Sensibilità / TPR
    specificity: float       # TNR
    f1: float
    auc: float

class RandomForestModel:
    """
    Classe che gestisce:
    - Addestramento di un RandomForestClassifier
    - Validazione incrociata stratificata (Stratified K-Fold)
    - Calcolo di varie metriche per ogni fold e in media
    """
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int | None = None,
        n_jobs: int = -1,
        save_csv: bool = False,
        csv_path: str = "metrics_per_fold.csv",
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.save_csv = save_csv
        self.csv_path = csv_path

    @staticmethod
    def _to_numpy_xy(X, y) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converte X e y in array NumPy e trasforma il target in binario {0,1}.
        Gestisce booleani, 0/1, stringhe "true/false" e classi testuali.
        """
        X_np = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.array(X)
        y_np = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else np.array(y)
        y_np = y_np.ravel()

        # Se il target è booleano → cast diretto
        if y_np.dtype == bool:
            y_bin = y_np.astype(int)
        else:
            y_series = pd.Series(y_np)
            # Se è già {0,1}
            if set(y_series.unique()) == {0, 1}:
                y_bin = y_np.astype(int)
            else:
                # Gestione stringhe "true"/"false"
                y_lower = y_series.astype(str).str.lower()
                if set(y_lower.unique()).issubset({"true", "false"}):
                    y_bin = (y_lower == "true").astype(int).to_numpy()
                else:
                    # Ordina le classi e considera la più "alta" come positiva
                    classes_sorted = sorted(y_series.unique())
                    pos_class = classes_sorted[-1]
                    y_bin = (y_series == pos_class).astype(int).to_numpy()

        return X_np, y_bin

    def train_and_evaluate(self, X, y) -> Dict[str, Any]:
        """
        Esegue Stratified K-Fold CV su RandomForestClassifier,
        calcola metriche per fold e metriche medie finali.
        """
        X_np, y_np = self._to_numpy_xy(X, y)

        # StratifiedKFold mantiene la proporzione delle classi in ogni fold
        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        fold_results: list[FoldResult] = []
        global_cm = np.zeros((2, 2), dtype=int)  # matrice di confusione aggregata

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_np, y_np), start=1):
            X_tr, X_va = X_np[tr_idx], X_np[va_idx]
            y_tr, y_va = y_np[tr_idx], y_np[va_idx]

            # Crea e addestra RandomForest
            clf = RandomForestClassifier(**self.model_params)
            clf.fit(X_tr, y_tr)

            # Predizioni e probabilità
            y_pred = clf.predict(X_va)
            y_prob = clf.predict_proba(X_va)[:, 1]  # Prob. classe positiva

            # Calcolo confusion matrix (TN, FP, FN, TP)
            tn, fp, fn, tp = confusion_matrix(y_va, y_pred, labels=[0, 1]).ravel()

            # Metriche standard con sklearn
            accuracy = accuracy_score(y_va, y_pred)
            error_rate = 1.0 - accuracy
            precision = precision_score(y_va, y_pred, zero_division=0)
            recall = recall_score(y_va, y_pred, zero_division=0)  # Sensibilità / TPR
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_va, y_pred, zero_division=0)

            # Calcolo AUC (gestione caso di classi uniche)
            try:
                auc = roc_auc_score(y_va, y_prob)
            except ValueError:
                auc = np.nan

            # Aggiorna matrice globale
            global_cm += np.array([[tn, fp], [fn, tp]], dtype=int)

            # Salva risultati del fold
            fr = FoldResult(
                fold=fold_idx,
                tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=accuracy,
                error_rate=error_rate,
                precision=precision,
                recall=recall,
                specificity=specificity,
                f1=f1,
                auc=auc,
            )
            fold_results.append(fr)

            # Stampa risultati del fold
            print(f"\n--- Fold {fold_idx} ---")
            print(f"Confusion Matrix:\n[[{tn} {fp}]\n [{fn} {tp}]]")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Error Rate: {error_rate:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Sensitivity (Recall): {recall:.4f}")
            print(f"Specificity: {specificity:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"AUC: {auc:.4f}" if not np.isnan(auc) else "AUC: N/A")

        # DataFrame con risultati per fold
        df = pd.DataFrame([{
            "Fold": fr.fold,
            "TN": fr.tn, "FP": fr.fp, "FN": fr.fn, "TP": fr.tp,
            "Accuracy": fr.accuracy,
            "Error Rate": fr.error_rate,
            "Precision": fr.precision,
            "Recall": fr.recall,
            "Specificity": fr.specificity,
            "F1": fr.f1,
            "AUC": fr.auc,
        } for fr in fold_results])

        # Salvataggio opzionale su CSV
        if self.save_csv:
            df.to_csv(self.csv_path, index=False)
            print(f"\nSalvato report per fold in: {self.csv_path}")

        # Calcolo medie metriche + deviazione standard dell'accuratezza
        mean_metrics = {
            "Confusion Matrix (sum)": global_cm,
            "Accuracy": float(df["Accuracy"].mean()),
            "Accuracy StdDev": float(df["Accuracy"].std()),  # deviazione standard
            "Error Rate": float(df["Error Rate"].mean()),
            "Precision": float(df["Precision"].mean()),
            "Sensitivity (Recall)": float(df["Recall"].mean()),
            "Specificity": float(df["Specificity"].mean()),
            "F1": float(df["F1"].mean()),
            "AUC": float(np.nanmean(df["AUC"].to_numpy())),
        }

        return {
            "per_fold": df,               # DataFrame con metriche per ogni fold
            "mean_metrics": mean_metrics, # Medie delle metriche
            "global_confusion": global_cm # Matrice di confusione aggregata
        }
