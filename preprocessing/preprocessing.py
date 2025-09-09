import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer


def encode_booleans(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = ['Transported', 'VIP', 'CryoSleep']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0})
    return df


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    group_counts = df['Group'].value_counts()
    df['Group_size'] = df['Group'].map(group_counts)
    return df


def add_lastname_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['Lastname'] = df['Name'].str.split().str[-1]
    return df


def add_cabin_features(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    if 'Cabin' not in df.columns:
        return df

    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0].fillna("Unknown")
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors="coerce")
    df['Side'] = cabin_split[2].fillna("Unknown")

    if df['CabinNum'].notna().sum() > 0:
        try:
            df['Cabin_region'] = pd.qcut(df['CabinNum'], q=4, duplicates="drop")
        except ValueError:
            df['Cabin_region'] = pd.cut(df['CabinNum'], bins=range(0, 2000, 300))
    else:
        df['Cabin_region'] = "Unknown"

    if drop_original:
        df = df.drop(columns=['Cabin'])
    return df


def add_expense_features(df: pd.DataFrame) -> pd.DataFrame:
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[expense_cols] = df[expense_cols].fillna(0)

    for col in expense_cols:
        df[f'Log_{col}'] = np.log1p(df[col])

    df['Expenditures'] = df[[f'Log_{c}' for c in expense_cols]].sum(axis=1)
    df = df.drop(columns=expense_cols)
    return df


def knn_impute(df: pd.DataFrame, discrete_cols: list = None, n_neighbors: int = 15) -> pd.DataFrame:
    df = df.copy().replace({pd.NA: np.nan})
    exclude_cols = ["Transported", "PassengerId"]

    dummy_cols = [
        c for c in df.columns
        if c not in exclude_cols
           and df[c].dropna().isin([0, 1]).all()
           and df[c].dtype in ["int64", "float64"]
    ]

    numeric_cols = (
        df.drop(columns=exclude_cols + dummy_cols, errors="ignore")
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

    mask_nan = df[numeric_cols].isna()
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols].dropna())
    df_scaled_numeric = pd.DataFrame(
        scaler.transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index
    )

    df_for_impute = pd.concat([df_scaled_numeric, df[dummy_cols]], axis=1)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_for_impute), columns=df_for_impute.columns,
                                     index=df.index)

    df_imputed_numeric = df[numeric_cols].copy()
    for col in numeric_cols:
        rows_nan = mask_nan[col]
        if rows_nan.any():
            vals_scaled = df_imputed_scaled.loc[rows_nan, col].values.reshape(-1, 1)
            vals_orig_scale = scaler.inverse_transform(
                np.column_stack([vals_scaled if c == col else np.zeros_like(vals_scaled) for c in numeric_cols])
            )[:, numeric_cols.index(col)]
            df_imputed_numeric.loc[rows_nan, col] = vals_orig_scale

    df_final = df.copy()
    df_final[numeric_cols] = df_imputed_numeric

    if discrete_cols is not None:
        for col in discrete_cols:
            if col in df_final.columns:
                df_final[col] = df_final[col].round().astype("Int64")
    return df_final


def encode_categoricals(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    categorical = ['Deck', 'HomePlanet', 'Destination', 'Side', 'Cabin_region']
    df[categorical] = df[categorical].fillna("Unknown")

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(df.loc[train_mask, categorical])

    encoded_array = encoder.transform(df[categorical])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical), index=df.index)

    df = df.drop(columns=categorical)
    df = pd.concat([df, encoded_df], axis=1)
    return df


def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df = df.drop(columns=to_drop)
    return df


def preprocess_dataset(df: pd.DataFrame, train_mask: pd.Series = None) -> pd.DataFrame:
    df = encode_booleans(df)
    df = add_group_features(df)
    df = add_lastname_feature(df)
    df = add_cabin_features(df)
    df = add_expense_features(df)

    # Imputazione KNN (numeriche + dummy)
    numeric_discrete_cols = ['Group_size', 'CabinNum']
    df = knn_impute(df, discrete_cols=numeric_discrete_cols)

    # Codifica categoriche con OHE
    if train_mask is not None:
        df = encode_categoricals(df, train_mask)

    # Rimuovo feature altamente correlate
    df = remove_highly_correlated(df, threshold=0.9)

    return df