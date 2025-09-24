from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configurazione base per RandomForest + K-Fold."""
    n_splits: int = 10
    random_state: int = 42
    shuffle: bool = True
    # RandomForest
    n_estimators: int = 500
    max_depth: Optional[int] = None
    max_features: str = "sqrt"
    n_jobs: int = -1
    class_weight: Optional[str] = None
    # colonne da escludere
    drop_cols: Tuple[str, ...] = ("PassengerId", "Name", "Cabin")


class RandomForestCV:
    """
    - Allena RF con StratifiedKFold
    - Tiene le proba OOF
    - Espone un unico riassunto: accuracy media, std, confusion matrix OOF
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.models_: List[Pipeline] = []
        self.oof_proba_: Optional[np.ndarray] = None
        self.fold_indices_: List[np.ndarray] = []
        logger.info(
            "RandomForestCV inizializzato | n_splits=%d, RF(n_estimators=%d, max_depth=%s, max_features=%s)",
            self.cfg.n_splits, self.cfg.n_estimators, str(self.cfg.max_depth), str(self.cfg.max_features)
        )

    # --- utils ---
    def _split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if "Transported" not in df.columns:
            raise ValueError("Manca la colonna target 'Transported'.")
        y = df["Transported"].astype(int)
        X = df.drop(columns=["Transported"])
        drop = [c for c in self.cfg.drop_cols if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
        logger.info("Split X/y effettuato | X shape=%s, y len=%d", tuple(X.shape), len(y))
        return X, y

    def _make_pipeline(self) -> Pipeline:

        rf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            max_features=self.cfg.max_features,
            n_jobs=self.cfg.n_jobs,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
        )
        return Pipeline([("rf", rf)])

    # --- training ---
    def fit(self, df: pd.DataFrame) -> "RandomForestCV":
        logger.info("Inizio training con StratifiedKFold (%d splits)...", self.cfg.n_splits)
        X, y = self._split_X_y(df)
        skf = StratifiedKFold(
            n_splits=self.cfg.n_splits,
            shuffle=self.cfg.shuffle,
            random_state=self.cfg.random_state
        )

        self.models_.clear()
        self.fold_indices_.clear()
        self.oof_proba_ = np.zeros(len(X), dtype=float)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        for tr_idx, va_idx in skf.split(X, y):
            logger.info("Training fold")
            # Standardizzazione training set
            scaler = StandardScaler()
            X_train_scaled = X.iloc[tr_idx].copy()
            X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_scaled[numeric_cols])

            # Standardizzazione validation set con media/varianza del training set
            X_valid_scaled = X.iloc[va_idx].copy()
            X_valid_scaled[numeric_cols] = scaler.transform(X_valid_scaled[numeric_cols])

            # Creazione pipeline e training
            pipe = self._make_pipeline()
            pipe.fit(X_train_scaled, y.iloc[tr_idx])

            # Predizione sul validation set
            proba = pipe.predict_proba(X_valid_scaled)[:, 1]

            # Salvataggio risultati
            self.oof_proba_[va_idx] = proba
            self.models_.append(pipe)
            self.fold_indices_.append(va_idx)

        return self

    # --- riassunto richiesto ---
    def summary_metrics(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        if self.oof_proba_ is None or not self.fold_indices_:
            raise RuntimeError("Chiama fit() prima.")

        logger.info("Calcolo metriche OOF con threshold=%.2f", threshold)
        _, y = self._split_X_y(df)
        y = y.values

        fold_acc = []
        for va_idx in self.fold_indices_:
            y_hat_fold = (self.oof_proba_[va_idx] >= threshold).astype(int)
            acc_fold = (y[va_idx] == y_hat_fold).mean()
            fold_acc.append(acc_fold)

        acc_mean = float(np.mean(fold_acc))
        acc_std = float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0
        tn, fp, fn, tp = confusion_matrix(y, (self.oof_proba_ >= threshold).astype(int)).ravel()

        # precision
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0

        logger.info("Accuracy media=%.4f ± %.4f | Confusion matrix tn=%d fp=%d fn=%d tp=%d",
                    acc_mean, acc_std, tn, fp, fn, tp)

        return {
            "n_splits": self.cfg.n_splits,
            "threshold": threshold,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "precision": precision,
            "specificity": specificity,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    # --- hyperparameter tuning ---
    def tune_hyperparameters(
            self,
            df: pd.DataFrame,
            param_grid: Optional[Dict[str, list]] = None,
            cv: int = 3,
            scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        X, y = self._split_X_y(df)

        if param_grid is None:
            param_grid = {
                "rf__n_estimators": [100, 300, 500],
                "rf__max_depth": [None, 10, 20, 30],
                "rf__max_features": ["sqrt", "log2", 0.5],
                "rf__class_weight": [None, "balanced"]
            }

        logger.info("Avvio GridSearchCV con %d combinazioni parametri, cv=%d",
                    int(np.prod([len(v) for v in param_grid.values()])), cv)

        # Creiamo una pipeline temporanea con StandardScaler
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # ColumnTransformer per scalare solo le colonne numeriche
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols)
            ],
            remainder="passthrough"  # lascia inalterate le altre colonne
        )

        # Pipeline completa: preprocessor + RandomForest
        rf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            max_features=self.cfg.max_features,
            n_jobs=self.cfg.n_jobs,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
        )
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("rf", rf)
        ])

        # GridSearchCV
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
        grid.fit(X, y)

        # Aggiorna self.cfg solo con i parametri trovati
        best_params = grid.best_params_
        for key, val in best_params.items():
            if key.startswith("rf__"):
                setattr(self.cfg, key[4:], val)

        logger.info("Migliori iperparametri trovati: %s", best_params)
        return best_params
