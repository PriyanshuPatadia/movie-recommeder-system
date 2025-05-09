import pickle
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Dummy example – replace with your actual movie data loading
movies_df = pd.read_csv('your_movies_data.csv')  # Replace with your CSV or source

# Replace this logic with your real processing pipeline
movies_df['combined_features'] = movies_df['genres'] + ' ' + movies_df['keywords'] + ' ' + movies_df['cast'] + ' ' + movies_df['director']

cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(movies_df['combined_features'])

similarity = cosine_similarity(count_matrix)

# Ensure model folder exists
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Save files
with open(os.path.join(model_dir, 'movie_list.pkl'), 'wb') as f:
    pickle.dump(movies_df, f)

with open(os.path.join(model_dir, 'similarity.pkl'), 'wb') as f:
    pickle.dump(similarity, f)

print("✅ Model files saved in 'model/' directory.")
