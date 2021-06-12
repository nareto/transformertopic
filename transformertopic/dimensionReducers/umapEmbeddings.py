import umap.umap_ as umap
from loguru import logger

class UmapEmbeddings():
    def __init__(self, n_components = 76, umapNNeighbors=15):
        """
        umapNNeighbors: umap parameter. Higher => fewer topics.
        umapNComponents: umap parameter. Higher => higher clustering accuracy but slower.
        """
        self.umapNNeighbors = umapNNeighbors
        self.n_components = n_components
        self.umapMetric = 'cosine'
        self.model = umap.UMAP(n_neighbors=self.umapNNeighbors,
                                n_components=self.n_components,
                                metric=self.umapMetric)
        self.model2d = umap.UMAP(n_neighbors=self.umapNNeighbors,
                                n_components=2,
                                metric=self.umapMetric)
                                

    def fit_transform(self, vectors):
        logger.debug(
            f"computing Umap embeddings with n_components = {self.n_components}, n_neighbors = {self.umapNNeighbors}, metric = {self.umapMetric}"
            )
        return self.model.fit_transform(vectors)

    def fit_transform2d(self, vectors):
        return self.model2d.fit_transform(vectors)

