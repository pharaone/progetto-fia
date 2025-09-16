# adaboost_kfold_min_logged.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging
import time
import warnings

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class ConfigAdaBoost:
    """Configurazione base per AdaBoost + K-Fold con logging."""
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True
    # Parametri AdaBoost
    n_estimators: int = 300
    learning_rate: float = 0.1
    algorithm: str = "SAMME"  # probabilità più stabili
    base_max_depth: int = 3     # stump (classico per AdaBoost)
    # colonne da escludere (ID, testo libero, potenziale leakage)
    drop_cols: Tuple[str, ...] = ("PassengerId", "Name", "Cabin")
    # logging
    log_level: int = logging.INFO  # usa logging.DEBUG per massima verbosità

class AdaBoostCV:
    """
    Wrapper compatto che:
    - allena un AdaBoost con StratifiedKFold
    - calcola metriche OOF (out-of-fold)
    - permette di tracciare curve ROC e PR
    - espone predict_proba / predict
    - LOGGA ogni step per diagnosticare eventuali blocchi
    """

    def __init__(self, cfg: Optional[ConfigAdaBoost] = None):
        self.cfg = cfg or ConfigAdaBoost()
        self.models_: List[Pipeline] = []
        self.oof_proba_: Optional[np.ndarray] = None
        self.fold_indices_: List[np.ndarray] = []
        # configura logging di default sulla base della cfg
        self.enable_logging(self.cfg.log_level)

        logger.debug("AdaBoostCV inizializzato con config: %s", self.cfg)

    # ------- Utils logging -------
    @staticmethod
    def enable_logging(level: int = logging.INFO):
        """Configura il root logger se non già configurato."""
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(
                level=level,
                format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            )
        else:
            root.setLevel(level)
        logger.setLevel(level)

    @staticmethod
    def _class_balance(y: pd.Series) -> Dict[str, float]:
        vc = y.value_counts(normalize=True).to_dict()
        return {str(k): float(v) for k, v in vc.items()}

    # ---------------- SPLIT ----------------
    def _split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.debug("Eseguo split X/y. Colonne disponibili: %s", list(df.columns))
        if "Transported" not in df.columns:
            logger.error("Manca la colonna target 'Transported'.")
            raise ValueError("Manca la colonna target 'Transported'.")
        X = df.drop(columns=["Transported"]).copy()
        y = df["Transported"].astype(int)
        drop = [c for c in self.cfg.drop_cols if c in X.columns]
        if drop:
            logger.info("Droppo colonne: %s", drop)
            X = X.drop(columns=drop)
        logger.info("Split completato: X shape=%s, y shape=%s", X.shape, y.shape)
        logger.debug("Bilanciamento classi (y): %s", self._class_balance(y))
        return X, y

    # ---------------- PIPELINE ----------------
    def _make_pipeline(self, X: pd.DataFrame) -> Pipeline:
        logger.debug("Creo pipeline. Dtypes numerici/categorici in X: %s",
                     X.dtypes.value_counts().to_dict())
        num_sel = selector(dtype_include=np.number)
        cat_sel = selector(dtype_include=object)

        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_sel),
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), cat_sel),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        base_tree = DecisionTreeClassifier(
            max_depth=self.cfg.base_max_depth,
            random_state=self.cfg.random_state
        )
        ada = AdaBoostClassifier(
            estimator=base_tree,
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            algorithm=self.cfg.algorithm,
            random_state=self.cfg.random_state
        )
        pipe = Pipeline([("pre", pre), ("clf", ada)])
        logger.debug("Pipeline creata.")
        return pipe

    # ---------------- TRAIN ----------------
    def fit(self, df: pd.DataFrame) -> "AdaBoostCV":
        t0 = time.perf_counter()
        logger.info("== Inizio fit AdaBoostCV ==")
        X, y = self._split_X_y(df)

        skf = StratifiedKFold(
            n_splits=self.cfg.n_splits,
            shuffle=self.cfg.shuffle,
            random_state=self.cfg.random_state
        )
        logger.info("StratifiedKFold: n_splits=%d, shuffle=%s, random_state=%s",
                    self.cfg.n_splits, self.cfg.shuffle, self.cfg.random_state)

        self.models_.clear()
        self.fold_indices_.clear()
        self.oof_proba_ = np.zeros(len(X), dtype=float)

        # Alcuni warning (es. feature costanti dopo OHE) possono confondere: li mostriamo come info
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
                logger.info("-- Fold %d/%d: train=%d, valid=%d",
                            fold, self.cfg.n_splits, len(tr_idx), len(va_idx))
                logger.debug("Class balance TRAIN: %s", self._class_balance(y.iloc[tr_idx]))
                logger.debug("Class balance VALID: %s", self._class_balance(y.iloc[va_idx]))

                try:
                    t_fold = time.perf_counter()
                    pipe = self._make_pipeline(X)
                    logger.debug("Inizio fit pipeline (fold %d)...", fold)
                    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                    logger.debug("Fine fit pipeline (fold %d) in %.3fs",
                                 fold, time.perf_counter() - t_fold)

                    logger.debug("Calcolo predict_proba (fold %d)...", fold)
                    proba = pipe.predict_proba(X.iloc[va_idx])[:, 1]
                    if np.any(~np.isfinite(proba)):
                        logger.warning("Probabilità non finite rilevate nel fold %d.", fold)

                    self.oof_proba_[va_idx] = proba
                    self.models_.append(pipe)
                    self.fold_indices_.append(va_idx)
                    logger.info("Fold %d completato in %.3fs", fold, time.perf_counter() - t_fold)

                except Exception as e:
                    logger.exception("Errore nel fold %d: %s", fold, str(e))
                    raise  # rialza per evidenziare esattamente il punto di rottura

            # Logghiamo eventuali warning catturati
            for w in wlist:
                logger.warning("Warning catturato durante fit: %s: %s", w.category.__name__, str(w.message))

        logger.info("== Fit completato in %.3fs ==", time.perf_counter() - t0)
        return self

    # ---------------- METRICHE ----------------
    @staticmethod
    def _metrics(y_true: np.ndarray, proba: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
        y_hat = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        out = {
            "threshold": thr,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "accuracy": float(accuracy_score(y_true, y_hat)),
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "f1": float(f1_score(y_true, y_hat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "pr_auc": float(average_precision_score(y_true, proba)),
        }
        logger.debug("Metriche calcolate: %s", out)
        return out

    def oof_metrics(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        if self.oof_proba_ is None:
            logger.error("oof_metrics chiamato prima di fit().")
            raise RuntimeError("Chiama fit() prima.")
        _, y = self._split_X_y(df)
        logger.info("Calcolo metriche OOF con threshold=%.3f", threshold)
        return self._metrics(y.values, self.oof_proba_, threshold)

    def per_fold_metrics(self, df: pd.DataFrame, threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.oof_proba_ is None or not self.fold_indices_:
            logger.error("per_fold_metrics chiamato prima di fit().")
            raise RuntimeError("Chiama fit() prima.")
        _, y = self._split_X_y(df)
        out = []
        for i, va_idx in enumerate(self.fold_indices_, 1):
            m = self._metrics(y.iloc[va_idx].values, self.oof_proba_[va_idx], threshold)
            m["fold"] = i
            m["support"] = int(len(va_idx))
            out.append(m)
        logger.info("Metriche per fold calcolate (%d folds).", len(out))
        return out

    # ---------------- CURVE ----------------
    def plot_oof_roc(self, df: pd.DataFrame, save_path: Optional[str] = None) -> float:
        if self.oof_proba_ is None:
            logger.error("plot_oof_roc chiamato prima di fit().")
            raise RuntimeError("Chiama fit() prima.")
        _, y = self._split_X_y(df)
        fpr, tpr, _ = roc_curve(y, self.oof_proba_)
        auc = roc_auc_score(y, self.oof_proba_)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC OOF - AdaBoost")
        plt.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info("ROC salvata in %s", save_path)
        plt.close()
        logger.debug("AUC ROC OOF=%.6f", auc)
        return float(auc)

    def plot_oof_pr(self, df: pd.DataFrame, save_path: Optional[str] = None) -> float:
        if self.oof_proba_ is None:
            logger.error("plot_oof_pr chiamato prima di fit().")
            raise RuntimeError("Chiama fit() prima.")
        _, y = self._split_X_y(df)
        prec, rec, _ = precision_recall_curve(y, self.oof_proba_)
        ap = average_precision_score(y, self.oof_proba_)
        plt.figure()
        plt.plot(rec, prec, label=f"AP={ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall OOF - AdaBoost")
        plt.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            logger.info("PR salvata in %s", save_path)
        plt.close()
        logger.debug("AP PR OOF=%.6f", ap)
        return float(ap)

    # ---------------- INFERENZA ----------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.models_:
            logger.error("predict_proba chiamato prima di fit().")
            raise RuntimeError("Chiama fit() prima.")
        X = df.drop(columns=[c for c in ("Transported",) if c in df.columns]).copy()
        drop = [c for c in self.cfg.drop_cols if c in X.columns]
        if drop:
            logger.info("Droppo colonne in inferenza: %s", drop)
            X = X.drop(columns=drop)
        logger.info("Inizio ensemble predict_proba su %d modelli. X shape=%s", len(self.models_), X.shape)
        t0 = time.perf_counter()
        probs = []
        for i, m in enumerate(self.models_, 1):
            t = time.perf_counter()
            p = m.predict_proba(X)[:, 1]
            probs.append(p)
            logger.debug("Model %d predict_proba completato in %.3fs", i, time.perf_counter() - t)
        out = np.mean(np.vstack(probs), axis=0)
        logger.info("Ensemble completato in %.3fs", time.perf_counter() - t0)
        return out

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        logger.info("Predict con threshold=%.3f", threshold)
        return (self.predict_proba(df) >= threshold).astype(int)
