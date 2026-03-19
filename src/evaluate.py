import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def evaluate_models(models, X_test, y_test):
    results = {}

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        print(f"\n{name} Accuracy: {acc}")
        print(f"{name} ROC-AUC: {roc_auc}")
        print(classification_report(y_test, preds))

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        print(f"{name} Confusion Matrix:\n{cm}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

        results[name] = roc_auc

    # Plot ROC Curve
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()

    return results