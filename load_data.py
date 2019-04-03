import pandas as pd
from os import path

_path = r"ml-1m"

def load(file):
    return pd.read_csv(path.join(_path, file), sep="::", usecols=[0, 1, 2], 
                names=["userID", "movieID", "rating"], engine="python")
    
movies  = load("movies.dat")
ratings = load("ratings.dat")
users   = load("users.dat")