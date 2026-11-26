import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle("sample_visual.pkl")  # optional sampled data

plt.figure(figsize=(8,5))
sns.histplot(df['TX_AMOUNT'], bins=50)
plt.title("Transaction Amount Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='TX_FRAUD', data=df)
plt.title("Fraud vs Non-Fraud")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='TX_FRAUD', y='TX_AMOUNT', data=df)
plt.title("Transaction Amount vs Fraud")
plt.show()
