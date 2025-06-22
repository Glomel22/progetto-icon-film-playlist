import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import os

def load_movielens_data(data_path):
    """
    Carica i dati MovieLens 100k
    """
    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(os.path.join(data_path, 'u.data'), 
                         sep='\t', names=ratings_cols)
    
    movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date',
                  'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                  "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                  'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    movies = pd.read_csv(os.path.join(data_path, 'u.item'), 
                        sep='|', names=movie_cols, encoding='latin-1')
    
    return ratings, movies

def preprocess_data(ratings, movies):
    """
    Preprocessing completo del dataset
    """
    avg_ratings = ratings.groupby('item_id')['rating'].agg(['mean', 'count']).reset_index()
    avg_ratings.columns = ['movie_id', 'avg_rating', 'rating_count']
    
    movies_data = movies.merge(avg_ratings, on='movie_id', how='inner')
    
    movies_data['release_year'] = pd.to_datetime(movies_data['release_date'], 
                                               errors='coerce').dt.year
    
    movies_data = movies_data.dropna(subset=['release_year'])
    movies_data = movies_data[movies_data['rating_count'] >= 5]
    
    genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                  'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
                  'Thriller', 'War', 'Western']
    
    final_data = movies_data[['movie_id', 'title', 'avg_rating', 'release_year'] + genre_cols].copy()
    
    scaler = MinMaxScaler()
    final_data['rating_norm'] = scaler.fit_transform(final_data[['avg_rating']])
    
    discretizer_rating = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    final_data['rating_discreto'] = discretizer_rating.fit_transform(final_data[['avg_rating']]).astype(int)
    
    discretizer_year = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    final_data['release_year_categoria'] = discretizer_year.fit_transform(final_data[['release_year']]).astype(int)
    
    return final_data, scaler, discretizer_rating, discretizer_year

def load_and_preprocess_data(dataset_path='data/ml-100k'):
    """
    Funzione orchestratrice che carica e preprocessa i dati.
    Questa è la funzione che main.py importerà.
    """
    print("Caricamento dati MovieLens 100k...")
    ratings, movies = load_movielens_data(dataset_path)
    
    print("Preprocessing dei dati...")
    processed_data, _, _, _ = preprocess_data(ratings, movies)
    
    return processed_data

def save_processed_data(data, output_path):
    """
    Salva i dati preprocessati
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Dati preprocessati salvati in: {output_path}")

def main():
    """
    Funzione principale eseguita solo quando questo script è lanciato direttamente.
    """
    output_path = 'data/processed_movies.csv'
    
    processed_data = load_and_preprocess_data()
    
    save_processed_data(processed_data, output_path)
    
    print(f"Dataset finale: {processed_data.shape[0]} film con {processed_data.shape[1]} feature")
    print("\nDistribuzione dei rating medi:")
    print(processed_data['avg_rating'].describe())
    
    return processed_data

def prepare_clustering_data(data):
    genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    clustering_features = genre_cols + ['rating_norm']
    return data[clustering_features], clustering_features

def prepare_classification_data(data):
    genre_cols = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    classification_features = genre_cols + ['rating_discreto']
    return data[classification_features], classification_features

def prepare_bayesian_data(data):
    main_genres = ['Action', 'Animation', 'Comedy', 'Drama', 'Thriller', 'Musical']
    bayesian_features = main_genres + ['rating_discreto', 'release_year_categoria', 'cluster']
    return data[bayesian_features], bayesian_features

if __name__ == "__main__":
    main()