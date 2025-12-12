import pandas as pd
df = pd.read_csv("/Users/utkarshsaxena/Desktop/college-student-burnout-analyzer/data/student_burnout_dataset.csv")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)
print("\nInfo:")
print(df.info())
