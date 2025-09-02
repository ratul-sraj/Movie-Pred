import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

st.title('Movie Recommendation System')

# --- Data Loading and Preprocessing ---
# Assuming your data files are in a directory named 'ml-32m'
# You might need to adjust the path based on where you store your data files
@st.cache_data # Cache the data loading to avoid reloading on every interaction
def load_data():
    data_path = 'ml-32m' # Adjust this path if necessary
    if not os.path.exists(data_path):
         st.error(f"Data directory not found at {data_path}. Please make sure the 'ml-32m' folder is in the same directory as the script.")
         return None, None, None, None

    try:
        df_ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
        df_tags = pd.read_csv(os.path.join(data_path, 'tags.csv'))
        df_movies = pd.read_csv(os.path.join(data_path, 'movies.csv'))
    except FileNotFoundError as e:
        st.error(f"Error loading data files. Make sure 'ratings.csv', 'tags.csv', and 'movies.csv' are in the '{data_path}' directory. {e}")
        return None, None, None, None


    # --- Feature Engineering ---
    movie_rated_mean = df_ratings.groupby("movieId").rating.mean().reset_index()
    df_movies = df_movies.merge(movie_rated_mean, on="movieId", how="left")

    movie_tags = df_tags.groupby('movieId')['tag'].apply(list).reset_index()
    df_movies = df_movies.merge(movie_tags, on='movieId', how='left')

    df_movies['tag'] = df_movies['tag'].fillna('').astype(str)
    df_movies['combined_features'] = df_movies['genres'] + ' ' + df_movies['tag']


    # --- TF-IDF Vectorization ---
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_movies['combined_features'].fillna('')) # Handle potential NaNs in combined_features


    # --- Cosine Similarity Calculation ---
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


    return df_movies, cosine_sim

df_movies, cosine_sim = load_data()

if df_movies is not None:
    # --- Recommendation Functions ---
    def find_movie_title(title):
        """
        Finds a movie title in the df_movies DataFrame by focusing on the core content of the title,
        ignoring information before or after it, and handling variations in spacing and capitalization.
        """
        def extract_core_title(t):
            match = re.search(r'^(.*?)\s*\(\d{4}\)', t)
            if match:
                return match.group(1).strip().lower()
            return t.strip().lower()

        cleaned_input_title = extract_core_title(title)

        for movie_title in df_movies['title']:
            if extract_core_title(movie_title) == cleaned_input_title:
                return movie_title
        return None


    def get_recommendations(title, cosine_sim=cosine_sim):
        """
        Gets movie recommendations based on cosine similarity.
        """
        if title not in df_movies['title'].values:
            return f"Movie with title '{title}' not found in the dataset."

        idx = df_movies[df_movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]

        return df_movies['title'].iloc[movie_indices].tolist()

    # --- Streamlit App Layout ---
    movie_title_input = st.text_input('Enter movie title:', placeholder='e.g., Toy Story')

    if st.button('Get Recommendations'):
        if movie_title_input:
            exact_title = find_movie_title(movie_title_input)
            if exact_title:
                recommendations = get_recommendations(exact_title, cosine_sim)
                if isinstance(recommendations, list):
                    st.subheader(f"Recommendations for '{exact_title}':")
                    for movie in recommendations:
                        st.write(f"- {movie}")
                else:
                    st.write(recommendations) # Display not found message
            else:
                st.write(f"Movie title '{movie_title_input}' not found or could not be matched.")
        else:
            st.write("Please enter a movie title.")

    # Optional: Display raw data or other info
    # if st.checkbox('Show raw data'):
    #     st.subheader('Movie Data')
    #     st.write(df_movies.head())
