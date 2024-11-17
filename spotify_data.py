import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import time

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-library-read,playlist-read-private"
))

playlist_ids = [
    "37i9dQZF1DX5wB72P2sVsT",   # Banger 
    "37i9dQZF1DWTIfBdh7WtFL",   # Nouveautés Electro
    "37i9dQZF1DX9ND1QF5hZNF",   # Electro Chill
    "37i9dQZF1DX4dyzvuaRJ0n",   # mint
    "37i9dQZF1DX7ZUug1ANKRP",   # Main Stage
    "37i9dQZF1DX0BcQWzuB7ZO",   # Dance Hits
    "37i9dQZF1DX12meDmp2yzx",   # Hit Dancefloor
    "37i9dQZF1DX8tZsk68tuDw",   # Dance Rising
    "37i9dQZF1DWZ7eJRBxKzdO",   # Summer Dance Hits 2024
    "37i9dQZF1DWWY64wDtewQt",   # Phonk
    "37i9dQZF1DWWn9pcJIAKFl",   # Tomorrowland Essentials
    "37i9dQZF1DXa3EUr6VSsMJ",   # French Touch
    "37i9dQZF1DX3d2wagqzwmM",   # Big Room Dance
    "37i9dQZF1DXaXB8fQg7xif",   # Dance Party
    "37i9dQZF1DX1zcQs2sF77z",   # Dancefloor Classic
]

def collect_playlist_tracks(playlist_id):
    results = sp.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

audio_features = []
track_ids_seen = set()

for playlist_id in playlist_ids:
    print(f"Data extraction for the playlist : {playlist_id}")
    tracks = collect_playlist_tracks(playlist_id)

    def is_popular_artist(artist_id):
        artist_info = sp.artist(artist_id)
        return artist_info['popularity'] > 80
    
    for item in tracks:
        track = item['track']
        if track and track['id'] not in track_ids_seen:
            track_id = track['id']
            artist_id = track['artists'][0]['id']
            track_ids_seen.add(track_id)
            try:
                features = sp.audio_features(track_id)[0]
                if features:
                    audio_features.append({
                        "name": track['name'],
                        "artist": track['artists'][0]['name'],
                        "genre": sp.artist(track['artists'][0]['id'])['genres'],
                        "release_date": track['album']['release_date'],
                        "is_popular_artist": is_popular_artist(artist_id),
                        "tempo": features['tempo'],
                        "energy": features['energy'],
                        "danceability": features['danceability'],
                        "valence": features['valence'],
                        "acousticness": features['acousticness'],
                        "speechiness": features['speechiness'],
                        "instrumentalness": features['instrumentalness'],
                        "liveness": features['liveness'],
                        "duration_ms": features['duration_ms'],
                        "popularity": track['popularity']
                    })
                time.sleep(0.1)
            except spotipy.exceptions.SpotifyException as e:
                print(f"Erreur avec le morceau {track_id}: {e}")
                time.sleep(1)


print(f"{len(audio_features)} collected tracks.")

df = pd.DataFrame(audio_features)
df.to_csv("spotify_extended_audio_features.csv", index=False)

print("Data extraction completed.")