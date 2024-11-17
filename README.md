# Spotify Popularity Prediction

This project aims to predict the popularity of electronic/dance tracks using various machine learning models (Linear Regression and XGBOOST). The pipeline includes data extraction from Spotify, data preprocessing, model training, and evaluation.

## Results

The results of the model training and evaluation are printed in the console. Visualizations are displayed to help understand the data and model performance.
For now, my model is not working well, I got bad result on the MSE and R² and I’m trying to figure out why, here are the leads :
- I don’t have enough data;
- Linear regression or xgboost may not be the right algorithms to use;
- the correlation between a song’s popularity and its metadata suggests a little more depth;

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scripts Overview](#scripts-overview)
- [Author](#author)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/iamhmh/ml_spotify_popularity_prediction
    cd ml_spotify_popularity_prediction
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up your Spotify API credentials in a `.env` file:
    ```properties
    CLIENT_ID=your_client_id
    CLIENT_SECRET=your_client_secret
    REDIRECT_URI=http://localhost:8888/callback
    ```

## Usage

To run the entire pipeline, execute the following command:
```sh
python main_pipeline.py
```

## Project Structure

```
ml_spotify_popularity_prediction/
│
├── data/
│   ├── spotify_audio_features.csv
│   ├── spotify_extended_audio_features.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│
├── scripts/
│   ├── spotify_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── exploration_data.py
│   ├── correlation_analysis.py
│   ├── main_pipeline.py
│
├── .gitignore
├── .env
├── README.md
├── requirements.txt
```

## Scripts Overview

### `spotify_data.py`
This script extracts audio features and metadata from various Spotify playlists and saves the data into a CSV file.

### `prepare_data.py`
This script preprocesses the data, performs feature engineering, and splits the data into training and testing sets.

### `train_model.py`
This script trains multiple machine learning models (Linear Regression, XGBoost) to predict the popularity of songs and evaluates their performance.

### `exploration_data.py`
This script provides visualizations to explore the distribution and relationships of different features in the dataset.

### `correlation_analysis.py`
This script generates a correlation matrix to analyze the relationships between numerical features.

### `main_pipeline.py`
This script orchestrates the execution of all other scripts in the correct order to ensure a smooth end-to-end pipeline.

## Author

- **HICHEM GOUIA** - (https://github.com/iamhmh)