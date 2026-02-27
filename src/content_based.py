from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_matrix(movies):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(movies["item_text"])

    return tfidf_matrix, vectorizer