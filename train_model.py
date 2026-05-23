import joblib
from sklearn.ensemble import RandomForestClassifier

# Load processed data
X_train_res, y_train_res, X_test, y_test = joblib.load("processed_data.pkl")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)

# Save model
joblib.dump(rf, "fraud_model.pkl")
print("Model trained and saved.")
