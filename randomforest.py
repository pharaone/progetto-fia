# random_forest_kfold_min.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

# librerie sklearn per ML e metriche
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

# per tracciare curve ROC / PR
import matplotlib.pyplot as plt


@dataclass
class Config:
    """Configurazione base per RF + K-Fold."""
    n_splits: int = 5            # numero di fold
    random_state: int = 42       # seed per riproducibilità
    shuffle: bool = True         # shuffle degli indici
    # Parametri della RandomForest
    n_estimators: int = 500
    max_depth: Optional[int] = None
    max_features: str = "sqrt"
    n_jobs: int = -1
    class_weight: Optional[str] = None
    # colonne da escludere (ID, testo libero, potenziale leakage)
    drop_cols: Tuple[str, ...] = ("PassengerId", "Name", "Cabin")


class RandomForestCV:
    """
    Wrapper compatto che:
    - allena una RandomForest con StratifiedKFold
    - calcola metriche OOF (out-of-fold)
    - permette di tracciare curve ROC e PR
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.models_: List[Pipeline] = []          # pipeline salvate per ogni fold
        self.oof_proba_: Optional[np.ndarray] = None  # predizioni probabilistiche OOF
        self.fold_indices_: List[np.ndarray] = []  # indici di validazione per fold

    # ---------------- SPLIT ----------------
    def _split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Divide X e y, rimuovendo colonne indesiderate."""
        if "Transported" not in df.columns:
            raise ValueError("Manca la colonna target 'Transported'.")
        X = df.drop(columns=["Transported"])
        y = df["Transported"].astype(int)
        # drop opzionali (PassengerId, ecc.)
        drop = [c for c in self.cfg.drop_cols if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
        return X, y

    # ---------------- PIPELINE ----------------
    def _make_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Pipeline: imputazione + OneHotEncoder + RandomForest."""
        # selettori automatici numeriche / categoriche
        num_sel = selector(dtype_include=np.number)
        cat_sel = selector(dtype_include=object)

        pre = ColumnTransformer(
            transformers=[
                # imputazione median per numeriche
                ("num", SimpleImputer(strategy="median"), num_sel),
                # imputazione most_frequent + one-hot per categoriche
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), cat_sel),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        rf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            max_features=self.cfg.max_features,
            n_jobs=self.cfg.n_jobs,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
        )
        return Pipeline([("pre", pre), ("rf", rf)])

    # ---------------- TRAIN ----------------
    def fit(self, df: pd.DataFrame) -> "RandomForestCV":
        """Addestra il modello con k-fold stratificato."""
        X, y = self._split_X_y(df)
        skf = StratifiedKFold(
            n_splits=self.cfg.n_splits, shuffle=self.cfg.shuffle, random_state=self.cfg.random_state
        )
        self.models_.clear()
        self.fold_indices_.clear()
        self.oof_proba_ = np.zeros(len(X), dtype=float)

        # ciclo sui fold
        for tr_idx, va_idx in skf.split(X, y):
            pipe = self._make_pipeline(X)
            pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])

            proba = pipe.predict_proba(X.iloc[va_idx])[:, 1]  # predizioni probabilistiche classe 1
            self.oof_proba_[va_idx] = proba
            self.models_.append(pipe)
            self.fold_indices_.append(va_idx)

        return self

    # ---------------- METRICHE ----------------
    @staticmethod
    def _metrics(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
        """Calcola metriche base e confusion matrix."""
        y_hat = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        return {
            "threshold": thr,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "accuracy": accuracy_score(y_true, y_hat),
            "precision": precision_score(y_true, y_hat, zero_division=0),
            "recall": recall_score(y_true, y_hat, zero_division=0),
            "f1": f1_score(y_true, y_hat, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba),
            "pr_auc": average_precision_score(y_true, proba),
        }

    def oof_metrics(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """Metriche su tutte le predizioni OOF."""
        if self.oof_proba_ is None:
            raise RuntimeError("Chiama fit() prima.")
        _, y = self._split_X_y(df)
        return self._metrics(y.values, self.oof_proba_, threshold)

    def per_fold_metrics(self, df: pd.DataFrame, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Metriche per singolo fold (usando indici validazione)."""
        if self.oof_proba_ is None or not self.fold_indices_:
            raise RuntimeError("Chiama fit() prima.")
        _, y = self._split_X_y(df)
        out = []
        for i, va_idx in enumerate(self.fold_indices_, 1):
            m = self._metrics(y.iloc[va_idx].values, self.oof_proba_[va_idx], threshold)
            m["fold"] = i
            m["support"] = int(len(va_idx))
            out.append(m)
        return out

    # ---------------- CURVE ----------------
    def plot_oof_roc(self, df: pd.DataFrame, save_path: Optional[str] = None) -> float:
        """Plotta ROC OOF e ritorna l’AUC."""
        if self.oof_proba_ is None:
            raise RuntimeError("Chiama fit() prima.")
        _, y = self._split_X_y(df)
        fpr, tpr, _ = roc_curve(y, self.oof_proba_)
        auc = roc_auc_score(y, self.oof_proba_)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC OOF - RandomForest")
        plt.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return float(auc)

    def plot_oof_pr(self, df: pd.DataFrame, save_path: Optional[str] = None) -> float:
        """Plotta Precision–Recall OOF e ritorna AP (average precision)."""
        if self.oof_proba_ is None:
            raise RuntimeError("Chiama fit() prima.")
        _, y = self._split_X_y(df)
        prec, rec, _ = precision_recall_curve(y, self.oof_proba_)
        ap = average_precision_score(y, self.oof_proba_)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall OOF - RandomForest")
        plt.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return float(ap)

    # ---------------- INFERENZA ----------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predizioni probabilistiche mediate tra i fold."""
        if not self.models_:
            raise RuntimeError("Chiama fit() prima.")
        X = df.drop(columns=[c for c in ("Transported",) if c in df.columns]).copy()
        drop = [c for c in self.cfg.drop_cols if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
        probs = [m.predict_proba(X)[:, 1] for m in self.models_]
        return np.mean(np.vstack(probs), axis=0)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predizione binaria con soglia (default 0.5)."""
        return (self.predict_proba(df) >= threshold).astype(int)
