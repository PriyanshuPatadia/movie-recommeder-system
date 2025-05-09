import pandas as pd
import numpy as np
import ast
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Initialize the stemmer
ps = PorterStemmer()

# Define functions for data preprocessing
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def collapse(L):
    return [i.replace(" ", "") for i in L]

def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

# Load datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on 'title'
movies = movies.merge(credits, on='title')

# Select relevant features
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values
movies.dropna(inplace=True)

# Apply preprocessing functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(fetch_director)

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create 'tags' feature
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
new_df['tags'] = new_df['tags'].apply(stem)

# Vectorize the 'tags' feature
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Compute cosine similarity matrix
similarity = cosine_similarity(vectors)

# Ensure 'model' directory exists
os.makedirs('model', exist_ok=True)

# Save the processed data and similarity matrix
pickle.dump(new_df, open('model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('model/similarity.pkl', 'wb'))

print("✅ 'movie_list.pkl' and 'similarity.pkl' have been successfully saved in the 'model/' directory.")
