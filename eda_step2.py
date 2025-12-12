import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/Users/utkarshsaxena/Desktop/student_burnout_dataset.csv")

# Encode burnout risk for color mapping
risk_map = {"Low": 0, "Medium": 1, "High": 2}
df["risk_encoded"] = df["burnout_risk"].map(risk_map)

# 🔹 Graph 1: Sleep vs Stress
plt.scatter(df["sleep_hours"], df["stress_level"], c=df["risk_encoded"])
plt.xlabel("Sleep Hours")
plt.ylabel("Stress Level")
plt.title("Sleep vs Stress (Burnout Risk)")
plt.show()

# 🔹 Graph 2: Screen Time vs Productivity
plt.scatter(df["screen_time"], df["productivity"], c=df["risk_encoded"])
plt.xlabel("Screen Time (hrs)")
plt.ylabel("Productivity Score")
plt.title("Screen Time vs Productivity (Burnout Risk)")
plt.show()
