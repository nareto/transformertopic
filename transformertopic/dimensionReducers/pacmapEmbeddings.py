import pacmap
from loguru import logger

class PacmapEmbeddings():
    def __init__(self, n_components=76, FP_ratio=2):
        # self.model = pacmap.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
        self.n_components = n_components
        self.model = pacmap.PaCMAP(n_dims = n_components, FP_ratio=FP_ratio)
        self.model2d = pacmap.PaCMAP(n_dims = 2)
                                
    def fit_transform(self, vectors):
        logger.debug(
            f"computing Pacmap embeddings with n_components = {self.n_components}"
            )
        return self.model.fit_transform(vectors)

    def fit_transform2d(self, vectors):
        return self.model2d.fit_transform(vectors)

