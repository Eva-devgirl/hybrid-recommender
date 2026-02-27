import cb

from src.data import load_movielens, create_item_text
from src.content_based import build_tfidf_matrix

if __name__ == "__main__":
    data_path = "ml-latest-small"
    
    ratings, movies = load_movielens(data_path)
    movies = create_item_text(movies)

    tfidf_matrix, vectorizer = build_tfidf_matrix(movies)
    print("\nTF-IDF matrix shape:", tfidf_matrix.shape)
    
    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    
    print("\nSample movie:")
    print(movies[["title", "genres", "item_text"]].head())


    #cb = ContentBasedRecommender()
    #cb.fit(movies)
    #print("\nTF-idf matrix shape:", cb.tfidf_matrix.shape)

    first_vector = tfidf_matrix[0].toarray()[0]

    print("\nLength of vectors:", len(first_vector))
    print("Non_zero elements:",(first_vector > 0).sum())
    #print("First 20 values:", first_vector[:20])

