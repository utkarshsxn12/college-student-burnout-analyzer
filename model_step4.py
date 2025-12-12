import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data
df = pd.read_csv("/Users/utkarshsaxena/Desktop/student_burnout_dataset.csv")

# Encode target
le = LabelEncoder()
df["burnout_risk"] = le.fit_transform(df["burnout_risk"])

X = df.drop("burnout_risk", axis=1)
y = df["burnout_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 🔹 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)
disp.plot()
plt.title("Confusion Matrix - Burnout Risk Prediction")
plt.show()

# 🔹 Feature Importance (coefficients)
importance = pd.Series(
    model.coef_[0], index=X.columns
).sort_values()

importance.plot(kind="barh")
plt.title("Feature Importance (Impact on Burnout Risk)")
plt.xlabel("Impact Weight")
plt.show()
