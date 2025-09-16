# main.py
import pandas as pd
from preprocessing import preprocess_dataset
from random_forest_kfold_min import RandomForestCV  # attenzione al nome file giusto!

def main():
    # 1) leggi l'input unico
    df = pd.read_csv("input.csv")

    # 2) separa train e test in base alla presenza di Transported
    train = df[df["Transported"].notna()].copy()
    test = df[df["Transported"].isna()].copy()

    # 3) preprocess
    train_p = preprocess_dataset(train)
    test_p = preprocess_dataset(test)

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

    # 7) predizioni sul test
    preds = model.predict(test_p, threshold=0.5)

    # 8) crea submission
    sub = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Transported": preds.astype(bool)
    })
    sub.to_csv("submission.csv", index=False)
    print("\nFile submission.csv creato con successo!")

if __name__ == "__main__":
    main()
