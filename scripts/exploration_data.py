import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("spotify_extended_audio_features.csv")

# Popularity Distribution Chart
plt.figure(figsize=(10, 6))
plt.hist(df['popularity'], bins=20, edgecolor='black')
plt.title("Popularity Distribution")
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.show()

# Acousticness and Popularity Relation Chart
plt.figure(figsize=(10, 6))
plt.scatter(df['tempo'], df['popularity'], alpha=0.7)
plt.title("Tempo and Popularity Relation")
plt.xlabel("Tempo")
plt.ylabel("Popularity")
plt.show()

# Energy and Popularity Relation Chart
plt.figure(figsize=(10, 6))
plt.scatter(df['energy'], df['popularity'], alpha=0.7, color='orange')
plt.title("Energy and Popularity Relation")
plt.xlabel("Energy")
plt.ylabel("Popularity")
plt.show()