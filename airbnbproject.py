import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data
df = pd.read_csv("airbnb_data.csv")

# Fill missing values
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'] = df['price'].fillna(0)
df['license'] = df['license'].fillna("Unlicensed")
df['number_of_reviews'] = df['number_of_reviews'].fillna(0)

# Basic info
print("Data Shape:", df.shape)
print(df.dtypes)

# Set plot style
sns.set(style="whitegrid")

# 1. Price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=50, kde=True, color='skyblue')
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.xlim(0, df['price'].quantile(0.95))  # Remove extreme outliers
plt.tight_layout()
plt.show()

# 2. Number of listings by license type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, y='license', order=df['license'].value_counts().index, palette="Set2")
plt.title("Listings by License Type")
plt.xlabel("Count")
plt.ylabel("License")
plt.tight_layout()
plt.show()

# 3. Average price by neighbourhood (top 10)
top_neigh = df.groupby("neighbourhood")["price"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_neigh.values, y=top_neigh.index, palette="Blues_r")
plt.title("Top 10 Neighbourhoods by Average Price")
plt.xlabel("Average Price")
plt.ylabel("Neighbourhood")
plt.tight_layout()
plt.show()

# 4. Relationship between reviews and price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='number_of_reviews', y='price', hue='license', alpha=0.6)
plt.title("Reviews vs. Price (Colored by License)")
plt.xlabel("Number of Reviews")
plt.ylabel("Price")
plt.ylim(0, df['price'].quantile(0.95))
plt.tight_layout()
plt.show()
