import pandas as pd
import os
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.externals import joblib

class initalizer:

    def __init__(self, items_csv_path="../data/new_movies.csv", ratings_csv_path="../data/ratings.csv"):
        self.path_movies = items_csv_path
        self.path_ratings = ratings_csv_path
        self.item_user_matrix_sparse, self.hashmap = self._prep_data()
        self.model = NearestNeighbors(n_neighbors=20, algorithm='brute', metric='cosine')
        self.model.fit(self.item_user_matrix_sparse)

    def _prep_data(self):

        #reading the data
        path = os.path.join(self.path_movies)
        usecols = ['movieId', 'tmdbId', 'title']
        dtype = {"movieId": 'int32', 'tmdbId': 'int32', "title": "str"}
        df_items = pd.read_csv(path, usecols=usecols, dtype=dtype)

        path = os.path.join(self.path_ratings)
        usecols = ['userId', 'movieId', 'rating']
        dtype = {"userId": 'int32', 'movieId': 'int32', "rating": "float32"}
        df_ratings = pd.read_csv(path, usecols=usecols, dtype=dtype)

        #pivot and create item-user matrix
        item_user_matrix = df_ratings.pivot(index='movieId', columns='userId', values='rating')
        #filling na's with 0's
        item_user_matrix = item_user_matrix.fillna(0)

        #transform the matrix into a scipy sparse matrix to minimize the nagitive impact
        # on calculation performances

        item_user_matrix_sparse = csr_matrix(item_user_matrix.values)

        #create mapper from item, the id to index of the movie in the item_user matrix
        hashmap = {
            item: i for i, item in
            enumerate(list(df_items.set_index('movieId').loc[item_user_matrix.index].tmdbId))
        }

        print(list(df_items.set_index('movieId').loc[item_user_matrix.index].tmdbId))
        print(df_items.set_index('movieId').loc[item_user_matrix.index].tmdbId)
        print(hashmap)
        print(hashmap.keys())
        print(hashmap[862.0])

        return item_user_matrix_sparse, hashmap

    def _save_data(self):
        """
        Save data to the disk
        1. hashmap
        2. scipy item-user sparse matrix
        3. trained model
        :return: none
        """
        np.save('./hashmap.npy', self.hashmap)
        save_npz("./matrix.npz", self.item_user_matrix_sparse)
        joblib.dump(self.model, './model.joblib')