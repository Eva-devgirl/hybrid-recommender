from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def build_surprise_dataset(ratings_df):
    """
    Convert a pandas DataFrame with columns  [userId, movieId, rating]
    into a Surprise Dataset object.
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
    return data

def train_svd(data):
    """
    Train an SVD collaborative filtering model.
    """

    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)

    algo = SVD()
    algo.fit(trainset)

    return algo