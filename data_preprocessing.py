import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
import pandas as pd

df = pd.read_csv(r"C:\Users\kusha\OneDrive\Desktop\kusha study\demo\creditcard.csv")
print(df.head())

X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Save processed data
joblib.dump((X_train_res, y_train_res, X_test, y_test), "processed_data.pkl")
print("Data preprocessing complete and saved.")
