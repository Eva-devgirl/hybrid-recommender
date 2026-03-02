from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_matrix(movies):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies["item_text"])

    return tfidf_matrix, vectorizer

def recommend_by_title(movies, tfidf_matrix, movie_title: str, top_k: int = 5):
    """
    Recommend top_k most similar movies to a given movie title based on cosine similarity.

    Parameters
    ----------
    movies : pd.DataFrame
        Must include columns: 'title', 'genres'.
        Row order must match tfidf_matrix row order.
    tfidf_matrix : scipy.sparse matrix
        TF-IDF vectors for movies (n_movies x vocab_size).
    movie_title : str
        Exact movie title to recommend.
    top_k : int
        Number of similar movies to recommend.

    Returns
    -------
    pd.DataFrame
        Recommended  movies with columns : 'title', 'genres', 'similarity'.
    """

    matches = movies.index[movies["title"] == movie_title].tolist()
    if not matches:
        raise ValueError(f"Movie title not found. {movie_title}")

    idx = matches[0]

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()
    sorted_idx = np.argsort(sims)[::-1]

    # Exclude itself (first one is the same movie with similarity 1.0)
    rec_idx = [i for i in sorted_idx if i != idx][:top_k]

    recs = movies.iloc[rec_idx][["title","genres"]].copy()
    recs["similarity"] = sims[rec_idx]
    recs = recs.reset_index(drop=True)

    return recs
