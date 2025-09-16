import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def knn_impute(df: pd.DataFrame, discrete_cols: list = None, n_neighbors: int = 15) -> pd.DataFrame:
    """
    Imputa i valori mancanti usando KNN Imputer con scaling.
    Versione per dataset non ancora diviso in train/val/test.

    Parametri
    ----------
    df : pd.DataFrame
        Dataset originale (senza colonne IsTrain/IsValidation/IsTest).
    discrete_cols : list
        Colonne numeriche da trattare come discrete (arrotondate dopo imputazione).
    n_neighbors : int
        Numero di vicini usato per KNN.

    Ritorna
    -------
    df_final : pd.DataFrame
        Dataset con valori mancanti imputati.
    """

    df = df.copy()
    df = df.replace({pd.NA: np.nan})

    # Colonne da escludere dall’imputazione
    exclude_cols = ["Transported", "PassengerId"]

    # Dummy binarie (0/1) → le includo in KNN ma non le scalo
    dummy_cols = [
        c for c in df.columns
        if c not in exclude_cols
        and df[c].dropna().isin([0, 1]).all()
        and df[c].dtype in ["int64", "float64"]
    ]

    # Numeriche continue da imputare
    numeric_cols = (
        df.drop(columns=exclude_cols + dummy_cols, errors="ignore")
          .select_dtypes(include=[np.number])
          .columns.tolist()
    )

    # Maschera dei NaN originali
    mask_nan = df[numeric_cols].isna()

    # Standardizza le numeriche
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols].dropna())
    df_scaled_numeric = pd.DataFrame(
        scaler.transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index
    )

    # Combina numeriche scalate + dummy intatte
    df_for_impute = pd.concat([df_scaled_numeric, df[dummy_cols]], axis=1)

    # Imputazione KNN
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_for_impute)
    df_imputed_scaled = pd.DataFrame(
        imputed_array, columns=df_for_impute.columns, index=df.index
    )

    # Ricostruisci numeriche riportandole alla scala originale
    df_imputed_numeric = df[numeric_cols].copy()
    for col in numeric_cols:
        rows_nan = mask_nan[col]
        if rows_nan.any():
            vals_scaled = df_imputed_scaled.loc[rows_nan, col].values.reshape(-1, 1)
            vals_orig_scale = scaler.inverse_transform(
                np.column_stack([
                    vals_scaled if c == col else np.zeros_like(vals_scaled)
                    for c in numeric_cols
                ])
            )[:, numeric_cols.index(col)]
            df_imputed_numeric.loc[rows_nan, col] = vals_orig_scale

    # Integra nel dataset originale
    df_final = df.copy()
    df_final[numeric_cols] = df_imputed_numeric

    # Arrotonda le colonne discrete, se specificate
    if discrete_cols is not None:
        for col in discrete_cols:
            if col in df_final.columns:
                df_final[col] = df_final[col].round().astype("Int64")

    return df_final