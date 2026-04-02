import pandas as pd
import numpy as np

# Data ingestion
df = pd.read_csv("student.csv")
print(df.head())
print(df.shape)

# Data Preprocessing

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Placement'] = le.fit_transform(df['Placement'])

# Dropping Unnecessary ID column
df = df.drop(columns=["Student ID"])

print(df.head())