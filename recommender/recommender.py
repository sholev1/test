from scipy.sparse import load_npz
import numpy as np
from sklearn.externals import joblib
class recommender:
    
    def __init__(self):
        """
        recommender requires a hashmap, item-user sparse matrix, and trained model parameters
        --------------------
        item_user_matrix_sparse: the item-user scipy sparse matrix
        hashmap: dict, maps item title name to index of the item in data
        model: NearestNeighbors model, trained model with data
        """
        self.item_user_matrix_sparse = load_npz('./matrix.npz')
        self.hashmap = np.load("./hashmap.npy")
        self.model = joblib.load('./model.joblib')


    def make_recommendations(self, itemId, n_recommendations):
        indexes = self._inference(itemId, n_recommendations)
        results = []
        reverse_hashmap = {v: k for k, v in self.hashmap.item().items()}
        for i in indexes[0]:
            results.append(reverse_hashmap[i])
        results.remove(itemId)
        return results

    def _inference(self, itemId, n_recomendations):
        itemIndex = self._find_item_index(itemId)
        distances, indexes = self.model.kneighbors(self.item_user_matrix_sparse[itemIndex], n_neighbors=n_recomendations +1)
        return indexes

    def _find_item_index(self, itemId):
        id = self.hashmap.item()[itemId]
        return id


# r = recommender()
# r = r.make_recommendations(13, 7)
#
# for i in r:
#     print(api)