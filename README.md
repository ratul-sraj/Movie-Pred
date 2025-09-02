# Movie Prediction Based on Cosine Similarity

This notebook demonstrates how to build a movie recommendation system using cosine similarity based on movie genres and user-provided tags.

## Data

The dataset used is the MovieLens 32M dataset, which includes movie ratings, tags, and movie metadata.

- `ratings.csv`: Contains user ratings for movies.
- `tags.csv`: Contains tags applied to movies by users.
- `movies.csv`: Contains movie titles and genres.

## Methodology

1.  **Data Loading and Preparation:**
    - Load the `ratings.csv`, `tags.csv`, and `movies.csv` files into pandas DataFrames.
    - Calculate the average rating for each movie from `ratings.csv`.
    - Merge the average ratings and tags into the `movies.csv` DataFrame.
    - Handle missing tags by replacing them with empty strings.

2.  **Feature Engineering:**
    - Create a combined feature string for each movie by concatenating its genres and tags.
    - Use TF-IDF (Term Frequency-Inverse Document Frequency) to convert the combined feature strings into a numerical matrix. This represents the importance of words (genres and tags) in each movie's feature set.

3.  **Similarity Calculation:**
    - Calculate the cosine similarity between all pairs of movies using the TF-IDF matrix. Cosine similarity measures the angle between two vectors, providing a score between 0 and 1, where 1 indicates perfect similarity.

4.  **Recommendation Function:**
    - Create a Python function `get_recommendations` that takes a movie title as input.
    - This function finds the input movie in the dataset, retrieves its corresponding row in the cosine similarity matrix, and identifies the movies with the highest similarity scores (excluding the input movie itself).
    - The function returns a list of the top N most similar movie titles.

5.  **Interactive Widget:**
    - An interactive widget is implemented using `ipywidgets` to allow users to enter a movie title and get real-time recommendations.
    - The widget also provides movie title suggestions as the user types.

## Key Findings

*   Missing values in the 'tag' column were successfully handled.
*   A combined feature representation using genres and tags was created.
*   TF-IDF was effectively used to vectorize the movie features.
*   Cosine similarity was calculated to measure the similarity between movies.
*   A recommendation function was developed to provide movie suggestions based on similarity.
*   An interactive widget was created for user-friendly recommendations.

## Code

The core logic is implemented in Python using libraries like `pandas` for data manipulation and `scikit-learn` for TF-IDF vectorization and cosine similarity calculation.
