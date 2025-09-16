# main.py
import pandas as pd
from preprocessing.preprocessing import preprocess_dataset
from randomforest import RandomForestCV


def main():
    # 1) leggi l'input unico
    df = pd.read_csv("input.csv")

    # 3) preprocess
    train_p = preprocess_dataset(df)

    # 4) addestra RandomForestCV
    model = RandomForestCV().fit(train_p)

    # 5) metriche OOF
    print("=== METRICHE OUT-OF-FOLD ===")
    print(model.oof_metrics(train_p, threshold=0.5))
    print("\n=== METRICHE PER FOLD ===")
    for m in model.per_fold_metrics(train_p, threshold=0.5):
        print(m)

    # 6) curve ROC/PR salvate su file
    print("\nSalvataggio grafici ROC/PR...")
    model.plot_oof_roc(train_p, save_path="roc_oof.png")
    model.plot_oof_pr(train_p, save_path="pr_oof.png")

if __name__ == "__main__":
    main()
