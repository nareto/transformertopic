from sklearn.manifold import TSNE
from loguru import logger

class TsneEmbeddings():
    def __init__(self, n_components=76):
        self.n_components = n_components
        self.model = TSNE(n_components=n_components, init='pca', method='exact')
        self.model2d = TSNE(n_components= 2, init ='pca', method='exact')
                                
    def fit_transform(self, vectors):
        logger.debug(
            f"computing Tsne embeddings with n_components = {self.n_components}"
            )
        return self.model.fit_transform(vectors)

    def fit_transform2d(self, vectors):
        return self.model2d.fit_transform(vectors)

