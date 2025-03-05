# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")  # Ensure train.csv is in the same folder

# Display the first 5 rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Visualize missing values
sns.heatmap(df.isnull(), cmap="coolwarm", cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

df.drop(columns=["Cabin"], inplace=True)

df["Age"].fillna(df["Age"].median(), inplace=True)

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
# Basic statistics of numerical columns
print(df.describe())

# Count of passengers by class
print(df["Pclass"].value_counts())

# Count of passengers by gender
print(df["Sex"].value_counts())

sns.countplot(x="Sex", hue="Survived", data=df, palette="Set2")
plt.title("Survival Rate by Gender")
plt.show()

sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

sns.barplot(x="Pclass", y="Survived", data=df, ci=None, palette="coolwarm")
plt.title("Survival Rate by Class")
plt.show()

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)