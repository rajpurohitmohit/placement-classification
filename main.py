import pandas as pd
import numpy as np

# Data ingestion
df = pd.read_csv("student.csv")
print(df.head())
print(df.shape)