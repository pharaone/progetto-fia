import pandas as pd

from preprocessing.preprocessing import preprocess_dataset


# Importa tutte le funzioni del preprocessing qui
# from preprocessing_module import preprocess_dataset  # se hai salvato il pipeline in un file separato

def main():
    # 1. Leggi dataset
    input_csv = "input.csv"   # sostituisci con il tuo path
    df = pd.read_csv(input_csv)

    # 2. Crea train_mask temporanea (necessaria per OHE)
    # Qui prendiamo tutti i dati come "train" per test
    train_mask = pd.Series([True] * len(df), index=df.index)

    # 3. Applica preprocessing
    df_processed = preprocess_dataset(df, train_mask=train_mask)

    # 4. Salva risultato
    output_csv = "preprocessed_output.csv"
    df_processed.to_csv(output_csv, index=False)
    print(f"Preprocessing completato. File salvato in: {output_csv}")

if __name__ == "__main__":
    main()
