import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


# EDA
# 1. Scatter Plot of IQ vs CGPA based on Placement
plt.figure(figsize=(8, 5))
sns.scatterplot(x='IQ', y='CGPA', hue='Placement', data=df, palette='coolwarm')
plt.title('IQ vs CGPA by Placement (1: Yes, 0: No)')
plt.savefig('scatter_iq_cgpa.png')
plt.show()

# 2. Box Plot of CGPA by Placement
plt.figure(figsize=(6, 4))
sns.boxplot(x='Placement', y='CGPA', data=df, palette='Set2')
plt.title('CGPA Distribution by Placement Status')
plt.savefig('boxplot_cgpa.png')
plt.show()

# 3. Box Plot of IQ by Placement
plt.figure(figsize=(6, 4))
sns.boxplot(x='Placement', y='IQ', data=df, palette='Set3')
plt.title('IQ Distribution by Placement Status')
plt.savefig('boxplot_iq.png')
plt.show()
