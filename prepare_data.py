import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("spotify_extended_audio_features.csv")

# Characteristic engineering and feature selection based on the correlation matrix
df['log_acousticness'] = np.log1p(df['acousticness'])
df['sqrt_instrumentalness'] = np.sqrt(df['instrumentalness'])
df['energy_danceability'] = df['energy'] * df['danceability']
df['acousticness_valence'] = df['acousticness'] * df['valence']
df['complexity'] = df['duration_ms'] * df['tempo']

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_date_days'] = (df['release_date'] - pd.Timestamp("1970-01-01")).dt.days

album_popularity = df.groupby('album_name')['popularity'].transform('mean')
df['album_popularity'] = album_popularity

df['valence_energy_ratio'] = df['valence'] / (df['energy'] + 0.01)
df['danceability_tempo'] = df['danceability'] * df['tempo']

df['genre_count'] = df['genre'].apply(lambda x: len(x) if isinstance(x, list) else 0)

df = df.dropna()  # Drop NaN values

X = df[['log_acousticness', 'sqrt_instrumentalness', 'energy_danceability', 
        'acousticness_valence', 'release_date_days', 'genre_count', 'complexity', 
        'album_popularity', 'valence_energy_ratio', 'danceability_tempo']]

y = df['popularity']

# Normalised data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Dividing the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Save the data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# Print the shapes of the data
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")