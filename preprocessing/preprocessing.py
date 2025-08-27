import numpy as np
import pandas as pd


def preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    #Creazione feature che classifica il gruppo a cui appartiene il passeggero a partire dall'id
    dataset['Group'] = dataset['PassengerId'].str.split('_').str[0].astype(int)

    dataset['Surname'] = dataset['Name'].str.split().str[-1]

    #Porto le variabili T/F in binario
    dataset['Transported'] = dataset['Transported'].map({True: 1, False: 0})
    dataset["Vip"] = dataset["Vip"].map({True: 1, False: 0})
    dataset["CryoSleep"] = dataset["CryoSleep"].map({True: 1, False: 0})

    expense_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    dataset[expense_columns] = dataset[expense_columns].fillna(0)

    # utlizzo la trasformazione con log(1 + x) per rendere utilizzabili i dati della spesa
    for column in expense_columns:
        dataset[f'new_{column}'] = np.log1p(dataset[column])

    # Rimuovo le colonne originali
    dataset = dataset.drop(columns=expense_columns)

    # Rimuovo transported per avere il training set pulito
    #dataset = dataset.drop(columns='Transported')
    return dataset





