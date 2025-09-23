import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from Adaboost import AdaBoostCV
from preprocessing.preprocessing import preprocess_dataset
from randomforest import RandomForestCV


def plot_confusion_from_dict(cm_dict, title, save_path):
    """
    cm_dict deve essere un dict con chiavi: tn, fp, fn, tp.
    Crea e salva un grafico di confusion matrix.
    """
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

    # 3) addestra i modelli
    rf = RandomForestCV().fit(dataset_p)
    ada = AdaBoostCV().fit(dataset_p)

    # 4) riassunti metriche (richiede che entrambe le classi abbiano summary_metrics)
    rf_sum = rf.summary_metrics(dataset_p, threshold=THRESHOLD)
    ada_sum = ada.summary_metrics(dataset_p, threshold=THRESHOLD)

    # 5) grafici confusion matrix
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

    # 6) CSV con accuracy mean/std
    rows = [
        {
            "model": "RandomForest",
            "n_splits": rf_sum["n_splits"],
            "threshold": rf_sum["threshold"],
            "accuracy_mean": rf_sum["accuracy_mean"],
            "accuracy_std": rf_sum["accuracy_std"],
        },
        {
            "model": "AdaBoost",
            "n_splits": ada_sum["n_splits"],
            "threshold": ada_sum["threshold"],
            "accuracy_mean": ada_sum["accuracy_mean"],
            "accuracy_std": ada_sum["accuracy_std"],
        },
    ]
    pd.DataFrame(rows).to_csv("metrics_summary.csv", index=False)

    print("Fatto!")
    print("Salvati: confusion_rf.png, confusion_adaboost.png, metrics_summary.csv")


if __name__ == "__main__":
    main()
