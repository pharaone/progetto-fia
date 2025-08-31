# main.py
import pandas as pd
from preprocessing import preprocess_dataset
from randomforest import RandomForestCV

def main():
    # 1) carica dataset
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # 2) preprocess
    train_p = preprocess_dataset(train)
    test_p = preprocess_dataset(test)  # attenzione: non contiene Transported

    # 3) addestra RandomForestCV
    model = RandomForestCV().fit(train_p)

    # 4) metriche OOF
    print("=== METRICHE OUT-OF-FOLD ===")
    print(model.oof_metrics(train_p, threshold=0.5))
    print("\n=== METRICHE PER FOLD ===")
    for m in model.per_fold_metrics(train_p, threshold=0.5):
        print(m)

    # 5) curve ROC/PR salvate su file
    print("\nSalvataggio grafici ROC/PR...")
    model.plot_oof_roc(train_p, save_path="roc_oof.png")
    model.plot_oof_pr(train_p, save_path="pr_oof.png")

    # 6) predizioni sul test
    preds = model.predict(test_p, threshold=0.5)

    # 7) crea submission
    sub = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Transported": preds.astype(bool)
    })
    sub.to_csv("submission.csv", index=False)
    print("\nFile submission.csv creato con successo!")

if __name__ == "__main__":
    main()
