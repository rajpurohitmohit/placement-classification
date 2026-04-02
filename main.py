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


# Feature Engineering & Feature Selection

# Feature Selection: with only 2 variables ('IQ', 'CGPA'), we don't need to drop any.
# Feature Engineering: We will standardize these numerical features as they have different scales.

X = df.drop(columns=['Placement'])
y = df['Placement']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Shapes post-splitting - X_train: {X_train.shape}, X_test: {X_test.shape}")

