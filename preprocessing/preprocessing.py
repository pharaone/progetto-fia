import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
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
    df['Name']= df['Name'].str.rsplit(n=1).str[0]
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

    df = df.drop(columns=['Deck', 'CabinNum', 'Side'])
    return df


def add_expense_features(df: pd.DataFrame) -> pd.DataFrame:
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[expense_cols] = df[expense_cols].fillna(0)

    for col in expense_cols:
        df[f'Log_{col}'] = np.log1p(df[col])

    df['Expenditures'] = df[[f'Log_{c}' for c in expense_cols]].sum(axis=1)
    df = df.drop(columns=expense_cols)
    return df


def knn_impute_sklearn(df: pd.DataFrame, n_neighbors: int = 15) -> pd.DataFrame:
    """
    Versione semplificata: usa direttamente sklearn.impute.KNNImputer
    """
    df = df.copy().replace({pd.NA: np.nan})
    exclude_cols = ["Transported", "PassengerId"]

    numeric_cols = df.drop(columns=exclude_cols, errors="ignore").select_dtypes(include=[np.number]).columns

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    categorical = ['HomePlanet', 'Destination', 'Cabin_region']

    for col in categorical:
        if col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].cat.add_categories("Unknown")
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].astype(str)

    # OneHotEncoder
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(df[categorical])

    encoded_array = encoder.transform(df[categorical])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical),
        index=df.index
    )

    df = df.drop(columns=categorical)
    df = pd.concat([df, encoded_df], axis=1)

    return df


def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.9,save_corr_plot: str = "correlation_matrix.png") -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr().abs()

    # Salva la heatmap della correlazione
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_corr_plot, dpi=300)
    plt.close()

    # Rimuovi feature troppo correlate
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    df_reduced = df.drop(columns=to_drop)

    print(f"Feature rimosse (correlazione > {threshold}): {to_drop}")
    return df_reduced



def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_booleans(df)
    df = add_group_features(df)
    df = add_cabin_features(df)
    df = add_expense_features(df)
    df = encode_categoricals(df)
    df = df.drop(columns=["PassengerId"])

    df = knn_impute_sklearn(df)
    # Rimuovo feature altamente correlate
    df = remove_highly_correlated(df, threshold=0.8)
    df = df.drop(columns=["Name"]) #verificato che non danno nulla

    return df