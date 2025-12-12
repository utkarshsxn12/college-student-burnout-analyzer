import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
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

# 🔹 User Input
print("Enter your daily details:")

study_hours = float(input("Study hours per day: "))
sleep_hours = float(input("Sleep hours per day: "))
screen_time = float(input("Screen time (hrs): "))
stress_level = int(input("Stress level (1–10): "))
caffeine_cups = int(input("Caffeine cups per day: "))
productivity = int(input("Productivity level (1–10): "))

user_data = [[
    study_hours,
    sleep_hours,
    screen_time,
    stress_level,
    caffeine_cups,
    productivity
]]

# Prediction
prediction = model.predict(user_data)
risk_label = le.inverse_transform(prediction)

print("\n🧠 Predicted Burnout Risk:", risk_label[0])
