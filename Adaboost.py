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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

@dataclass
class ConfigAdaBoost:
    """Configurazione base per AdaBoost + K-Fold con logging."""
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True
    # Parametri AdaBoost
    n_estimators: int = 150
    learning_rate: float = 0.1
    algorithm: str = "SAMME"   # probabilità più stabili in binario
    base_max_depth: int = 3    # stump (classico per AdaBoost)
    # colonne da escludere
    drop_cols: Tuple[str, ...] = ("PassengerId", "Name", "Cabin")
    # logging
    log_level: int = logging.INFO

class AdaBoostCV:
    """
    - Allena AdaBoost con StratifiedKFold
    - Tiene le proba OOF
    - Ritorna un riassunto: accuracy media, std, confusion matrix OOF
    """

    def __init__(self, cfg: Optional[ConfigAdaBoost] = None):
        self.cfg = cfg or ConfigAdaBoost()
        self.models_: List[Pipeline] = []
        self.oof_proba_: Optional[np.ndarray] = None
        self.fold_indices_: List[np.ndarray] = []
        self.enable_logging(self.cfg.log_level)
        logger.debug("AdaBoostCV inizializzato con config: %s", self.cfg)

    # ------- Utils logging -------
    @staticmethod
    def enable_logging(level: int = logging.INFO):
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
        logger.debug("Creo pipeline. Dtypes: %s", X.dtypes.value_counts().to_dict())
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

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
                logger.info("-- Fold %d/%d: train=%d, valid=%d",
                            fold, self.cfg.n_splits, len(tr_idx), len(va_idx))
                try:
                    t_fold = time.perf_counter()

                    # Standardizzazione training set
                    scaler = StandardScaler()
                    X_train_scaled = X.iloc[tr_idx].copy()
                    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_scaled[numeric_cols])

                    # Standardizzazione validation set con la stessa media/varianza del train
                    X_valid_scaled = X.iloc[va_idx].copy()
                    X_valid_scaled[numeric_cols] = scaler.transform(X_valid_scaled[numeric_cols])

                    # Creazione pipeline
                    pipe = self._make_pipeline(X_train_scaled)
                    pipe.fit(X_train_scaled, y.iloc[tr_idx])

                    # Predizione sul validation set
                    proba = pipe.predict_proba(X_valid_scaled)[:, 1]

                    if np.any(~np.isfinite(proba)):
                        logger.warning("Probabilità non finite rilevate nel fold %d.", fold)

                    self.oof_proba_[va_idx] = proba
                    self.models_.append(pipe)
                    self.fold_indices_.append(va_idx)
                    logger.info("Fold %d completato in %.3fs", fold, time.perf_counter() - t_fold)
                except Exception as e:
                    logger.exception("Errore nel fold %d: %s", fold, str(e))
                    raise

            for w in wlist:
                logger.warning("Warning catturato durante fit: %s: %s", w.category.__name__, str(w.message))

        logger.info("== Fit completato in %.3fs ==", time.perf_counter() - t0)
        return self
    # ---------------- RIASSUNTO METRICHE ----------------
    def summary_metrics(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Ritorna:
          - 'accuracy_mean': media dell'accuratezza sui fold
          - 'accuracy_std': deviazione standard campionaria (ddof=1) dell'accuratezza sui fold
          - 'confusion_matrix': dict {tn, fp, fn, tp} calcolata OOF con il threshold dato
        """
        if self.oof_proba_ is None or not self.fold_indices_:
            logger.error("summary_metrics chiamato prima di fit().")
            raise RuntimeError("Chiama fit() prima.")

        _, y = self._split_X_y(df)
        y = y.values

        # accuracy per fold
        fold_acc = []
        for va_idx in self.fold_indices_:
            y_hat_fold = (self.oof_proba_[va_idx] >= threshold).astype(int)
            acc_fold = (y[va_idx] == y_hat_fold).mean()
            fold_acc.append(acc_fold)

        acc_mean = float(np.mean(fold_acc))
        acc_std = float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0

        # confusion matrix OOF complessiva
        y_hat = (self.oof_proba_ >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()

        out = {
            "n_splits": self.cfg.n_splits,
            "threshold": threshold,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        logger.info("Summary metriche: %s", out)
        return out

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
