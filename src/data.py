import pandas as pd
from pathlib import Path

def load_movielens(data_path: str):
    """
    Loads Movielens small dataset.
    
    Parameters
    ----------
    data_path : str
    	Path to the ml-latest-small folder.
    	
    Returns
    -------
    ratings : pd.DataFrame
    movies : pd.DataFrame
    """
    
    data_path = Path(data_path)
    
    ratings = pd.read_csv(data_path / "ratings.csv")
    movies = pd.read_csv(data_path / "movies.csv")
    
    return ratings, movies
    
def create_item_text(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a text representation for each movie
    (used later for TF-IDF content-based model)
    """
    
    movies = movies.copy()
    
    movies["genres"] = movies["genres"].fillna("")
    movies["title"] = movies["title"].fillna("")
    
    movies["item_text"] = (
    	movies["title"].str.lower() + " " +
    	movies["genres"].str.replace("|", " ", regex=False).str.lower()
    )
    
    return movies


