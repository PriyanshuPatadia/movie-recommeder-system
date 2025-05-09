import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Example movie data (for simplicity, using a small sample)
movie_data = {
    'movie_id': [1, 2, 3, 4],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D'],
    'genre': ['Action', 'Comedy', 'Action', 'Comedy'],
    'ratings': [4.5, 3.7, 4.8, 3.2]
}

# Creating a DataFrame for the movies
movies_df = pd.DataFrame(movie_data)

# Saving the movie data (movies_.pickle)
with open('movies_.pickle', 'wb') as f:
    pickle.dump(movies_df, f)

# Example of movie feature vectors (could be from ratings, genres, etc.)
# Here we're simplifying and using just the ratings as features for similarity computation
movie_features = movies_df[['ratings']].values

# Standardizing the features (optional step depending on data)
scaler = StandardScaler()
movie_features_scaled = scaler.fit_transform(movie_features)

# Calculating the similarity matrix (using cosine similarity)
similarity_matrix = cosine_similarity(movie_features_scaled)

# Saving the similarity matrix (similarity.pickle)
with open('similarity.pickle', 'wb') as f:
    pickle.dump(similarity_matrix, f)

print("Pickle files saved: movies_.pickle and similarity.pickle")
