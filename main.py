import numpy as np
from src.data import load_movielens, create_item_text
from src.content_based import build_tfidf_matrix, recommend_by_title
from sklearn.metrics.pairwise import cosine_similarity
from src.collaborative import build_surprise_dataset
from src.collaborative import train_svd

if __name__ == "__main__":
    data_path = "ml-latest-small"
    
    ratings, movies = load_movielens(data_path)
    movies = create_item_text(movies)

    tfidf_matrix, vectorizer = build_tfidf_matrix(movies)
    print("\nTF-IDF matrix shape:", tfidf_matrix.shape)

    recs = recommend_by_title(movies, tfidf_matrix,"Toy Story (1995)", top_k = 5)
    print("\nRecommendations: for Toy Story (1995):")
    print(recs)

    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix).flatten()

    data = build_surprise_dataset(ratings)
    algo =train_svd(data)
    print("\nSurprise dataset built successfully.")

    prediction = algo.predict(uid=1, iid=1)
    print("\nPredicted rating for user 1 on movie 1:", prediction.est)
