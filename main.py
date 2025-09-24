import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from Adaboost import AdaBoostCV
from preprocessing.preprocessing import preprocess_dataset
from randomforest import RandomForestCV


def plot_confusion_from_dict(cm_dict, title, save_path):
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=int)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    THRESHOLD = 0.5

    # 1) leggi l'input unico
    df = pd.read_csv("input.csv")

    # 2) preprocess
    dataset_p = preprocess_dataset(df)
    print(dataset_p.head())
    dataset_p.to_csv('my_file.csv', index=False)

    # 3) Hyperparameter tuning RandomForest
    rf = RandomForestCV()
    print("Esecuzione tuning iperparametri RandomForest...")
    best_params_rf = rf.tune_hyperparameters(dataset_p, cv=3)
    print("Migliori parametri RandomForest:", best_params_rf)

    # 4) Fit finale sui fold
    print("Fit finale RandomForest con parametri ottimali...")
    rf.fit(dataset_p)

    # 5) Hyperparameter tuning AdaBoost
    ada = AdaBoostCV()
    print("Esecuzione tuning iperparametri AdaBoost...")
    best_params_ada = ada.tune_hyperparameters(dataset_p, cv=3)
    print("Migliori parametri AdaBoost:", best_params_ada)

    # 6) Fit finale sui fold
    print("Fit finale AdaBoost con parametri ottimali...")
    ada.fit(dataset_p)

    # 7) Riassunti metriche
    rf_sum = rf.summary_metrics(dataset_p, threshold=THRESHOLD)
    ada_sum = ada.summary_metrics(dataset_p, threshold=THRESHOLD)

    # 8) Grafici confusion matrix
    plot_confusion_from_dict(
        rf_sum["confusion_matrix"],
        title=f"Confusion Matrix - RandomForest (thr={THRESHOLD})",
        save_path="confusion_rf.png",
    )
    plot_confusion_from_dict(
        ada_sum["confusion_matrix"],
        title=f"Confusion Matrix - AdaBoost (thr={THRESHOLD})",
        save_path="confusion_adaboost.png",
    )

    # 9) CSV con accuracy mean/std/sem
    def _safe_sem(summary: dict) -> float:
        if "accuracy_sem" in summary and summary["accuracy_sem"] is not None:
            return float(summary["accuracy_sem"])
        n = int(summary.get("n_splits", 0))
        std = float(summary.get("accuracy_std", 0.0))
        return float(std / np.sqrt(n)) if n and n > 1 else 0.0

    rows = [
        {
            "model": "RandomForest",
            "n_splits": rf_sum["n_splits"],
            "threshold": rf_sum["threshold"],
            "accuracy_mean": rf_sum["accuracy_mean"],
            "accuracy_std": rf_sum["accuracy_std"],
            "accuracy_sem": _safe_sem(rf_sum),
        },
        {
            "model": "AdaBoost",
            "n_splits": ada_sum["n_splits"],
            "threshold": ada_sum["threshold"],
            "accuracy_mean": ada_sum["accuracy_mean"],
            "accuracy_std": ada_sum["accuracy_std"],
            "accuracy_sem": _safe_sem(ada_sum),
        },
    ]
    pd.DataFrame(rows).to_csv("metrics_summary.csv", index=False)

    print("Fatto!")
    print("Salvati: confusion_rf.png, confusion_adaboost.png, metrics_summary.csv")



if __name__ == "__main__":
    main()
