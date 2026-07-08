import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("train.csv")

# -----------------------------
# Display Basic Information
# -----------------------------
print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

# -----------------------------
# Data Cleaning
# -----------------------------

# Fill missing Age values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df.drop(columns=["Cabin"], inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

sns.set_style("whitegrid")

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.show()

# 2. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.show()

# 3. Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# 4. Passenger Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Pclass")
plt.title("Passenger Class Distribution")
plt.show()

# 5. Fare Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 7. Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.show()

# 8. Age vs Survival
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Survived", y="Age")
plt.title("Age vs Survival")
plt.show()

print("\nEDA Completed Successfully!")