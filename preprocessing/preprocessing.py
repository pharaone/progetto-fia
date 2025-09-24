import numpy as np
import pandas as pd
import seaborn as sns
import logging
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)

#procedo con l'encoding delle feature booleane
def encode_booleans(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding colonne booleane")
    bool_cols = ['Transported', 'VIP', 'CryoSleep']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0})
    return df

#ricavo dal passengerId il gruppo e la dimensione del gruppo del passeggero
def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggiunta feature di gruppo da PassengerId")
    df['Group'] = df['PassengerId'].str.split('_').str[0].astype(int)
    group_counts = df['Group'].value_counts()
    df['Group_size'] = df['Group'].map(group_counts)
    return df


#non usata più poichè non ha migliorato le prestazioni del modello
def add_lastname_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['Lastname'] = df['Name'].str.split().str[-1]
    df['Name']= df['Name'].str.rsplit(n=1).str[0]
    return df

#partendo dalla colonna cabin estraggo della feature di "zona"
def add_cabin_features(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    if 'Cabin' not in df.columns:
        return df
    logger.info("Estrazione feature dalla colonna Cabin")
    cabin_split = df['Cabin'].str.split('/', expand=True)
    df['Deck'] = cabin_split[0].fillna("Unknown")       #estraggo il deck
    df['CabinNum'] = pd.to_numeric(cabin_split[1], errors="coerce")     #estraggo il numero di cabina
    df['Side'] = cabin_split[2].fillna("Unknown")           #estraggo il lato

    if df['CabinNum'].notna().sum() > 0:
        try:
            #divido in 5 quantili per ridurre da num di cabina a regione
            df['Cabin_region'] = pd.qcut(df['CabinNum'], q=5, duplicates="drop")
        except ValueError:
            df['Cabin_region'] = pd.cut(df['CabinNum'], bins=range(0, 2000, 300))
    else:
        df['Cabin_region'] = "Unknown"

    if drop_original:
        df = df.drop(columns=['Cabin'])

    df = df.drop(columns=['Deck', 'CabinNum', 'Side'])
    return df

#aggiunga una feature che contiene la somma logaritmica di tutte le spese
def add_expense_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creazione feature somma delle spese")
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[expense_cols] = df[expense_cols].fillna(0)

    #faccio in questo modo per normalizzare valori sbilanciati
    for col in expense_cols:
        df[f'Log_{col}'] = np.log1p(df[col])

    df['Expenditures'] = df[[f'Log_{c}' for c in expense_cols]].sum(axis=1)
    df = df.drop(columns=expense_cols)
    return df

#uso la knn imputation per imputare gli eventuali campi mancanti delle colonne numeriche
def knn_impute_sklearn(df: pd.DataFrame, n_neighbors: int = 15) -> pd.DataFrame:
    logger.info(f"Imputazione valori mancanti con KNNImputer (n_neighbors={n_neighbors})")
    df = df.copy().replace({pd.NA: np.nan})
    exclude_cols = ["Transported", "PassengerId"]

    numeric_cols = df.drop(columns=exclude_cols, errors="ignore").select_dtypes(include=[np.number]).columns

    #il numero di vicini può essere configurato (default: 15)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Encoding delle variabili categoriche")

    # Colonne categoriche da codificare
    categorical = ['HomePlanet', 'Destination', 'Cabin_region']

    for col in categorical:
        if col in df.columns:
            # Se la colonna è categorica, aggiunge la categoria "Unknown"
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].cat.add_categories("Unknown")
            # Sostituisce i valori nulli con "Unknown"
            df[col] = df[col].fillna("Unknown")
            # Converte la colonna in stringa per il OneHotEncoder
            df[col] = df[col].astype(str)

    # Inizializza OneHotEncoder (drop='first' per evitare multicollinearità)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(df[categorical])

    # Applica l'encoder e costruisce un nuovo DataFrame con le colonne codificate
    encoded_array = encoder.transform(df[categorical])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical),
        index=df.index
    )

    # Rimuove le colonne originali categoriche
    df = df.drop(columns=categorical)
    # Aggiunge le colonne codificate
    df = pd.concat([df, encoded_df], axis=1)

    # Rimuove eventuali colonne "Unknown" mai attivate (tutti zeri)
    for col in encoded_df.columns:
        if "Unknown" in col and df[col].sum() == 0:
            df = df.drop(columns=[col])

    return df



#qui rimuoviamo una delle due colonne quando queste hanno hanno una correlazione fra loro maggiore di 0.9
def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.9,save_corr_plot: str = "correlation_matrix.png") -> pd.DataFrame:
    logger.info(f"Rimozione feature con correlazione > {threshold}")
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

    return df_reduced

#tutta la pipeline del preprocessing
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Inizio preprocessing del dataset...")
    df = encode_booleans(df)
    df = add_group_features(df)
    df = add_cabin_features(df)
    df = add_expense_features(df)

    df = knn_impute_sklearn(df)

    df = encode_categoricals(df)
    df = df.drop(columns=["PassengerId", "Name"], errors="ignore")

    df = remove_highly_correlated(df, threshold=0.8)
    logger.info("Preprocessing completato")
    return df