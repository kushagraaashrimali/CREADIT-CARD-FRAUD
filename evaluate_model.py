import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Load model and test data
rf = joblib.load("fraud_model.pkl")
_, _, X_test, y_test = joblib.load("processed_data.pkl")

# Precision-Recall curve
y_scores = rf.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

plt.plot(recall, precision, label=f"Random Forest (AUC={pr_auc:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()