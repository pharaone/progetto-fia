from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


@dataclass
class Config:
    """Configurazione base per RandomForest + K-Fold."""
    n_splits: int = 5
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

    # --- utils ---
    def _split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if "Transported" not in df.columns:
            raise ValueError("Manca la colonna target 'Transported'.")
        y = df["Transported"].astype(int)
        X = df.drop(columns=["Transported"])
        drop = [c for c in self.cfg.drop_cols if c in X.columns]
        if drop:
            X = X.drop(columns=drop)
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

        return {
            "n_splits": self.cfg.n_splits,
            "threshold": threshold,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
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

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])

        pipe = self._make_pipeline()

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
        grid.fit(X_scaled, y)

        best_params = grid.best_params_
        # aggiorna configurazione interna
        for key, val in best_params.items():
            if key.startswith("rf__"):
                setattr(self.cfg, key[4:], val)

        return best_params