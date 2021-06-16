import numpy as np
import spacy
from typing import Tuple, List

MIN_FLOAT = np.finfo('float').eps

def hash_array(array: np.array) -> str:
    return str([np.format_float_positional(comp) for comp in array])

class KMaxoids():
    def __init__(self, spacy_model="en_core_web_sm") -> None:
        self.spacy_model = spacy_model

    def fit_transform(self, documents, nKeywords=10) -> Tuple[List[str], List[np.array]]:
        text = ". ".join(documents)
        nlp = spacy.load(self.spacy_model)
        doc = nlp(text)
        filtered_doc = list(
            filter(
                lambda x: False if x.is_stop or x.is_punct else True, doc
            )
        )

        size_vec = filtered_doc[0].vector.shape[0]
        mat = np.zeros((size_vec, len(filtered_doc)))
        existing_vectors = set()
        for idx, d in enumerate(filtered_doc):
            vec = d.vector
            svec = hash_array(vec)
            while True:
                if svec in existing_vectors:
                    vec = vec + np.random.normal(0, 5*MIN_FLOAT, size=vec.shape)
                    svec = hash_array(vec)
                else:
                    existing_vectors.add(svec)
                    break
            mat[:, idx] = vec
        del existing_vectors
        kmaxoids = KMaxoidsClustering(mat, K=nKeywords)
        maxoids, labels = kmaxoids.run()
        keywords = []
        scores = []
        for idx,maxoid in enumerate(maxoids.T):
            col_vector = maxoid.reshape((len(maxoid), 1))
            distances = np.sum(((col_vector - mat)**2), axis=0)
            argmin = np.argmin(distances)
            keywords.append(str(filtered_doc[argmin]))
            scores.append(np.sum(labels == idx))
        return (keywords, scores)


class KMaxoidsClustering():
    def __init__(self, Y, K=2):
        """KMaxoids class. 

        Y: matrix where each column represents a data point
        K: number of desired clusters"""

        self.Y = Y
        self.dim, self.nsamples = Y.shape
        self.K = K
        if self.nsamples < self.K:
            self.K = self.nsamples


    def run(self, nruns=3, maxit=20):
        """Runs the Lloyd's algorithm variant described in [1] multiple times, each time with different random choices of initial maxoids. The final clusters are given by the run that gives best value of the optimization function. Returns tuple (maxoids,clusters) where:
        - maxoids is a matrix where each column is the reprensetative of the cluster
        - clusters is an array where each element is a set containing the index (column of self.Y) corresponding to the data point in that cluster

        Arguments:
        nruns: the number of times the algorithm is run 
        maxit: the number of iterations for each run

        [1]: http://ceur-ws.org/Vol-1458/E19_CRC4_Bauckhage.pdf"""

        best_val = np.infty
        for i in range(nruns):
            maxoids, labels = self._run_once(maxit)
            val = np.sum((self.Y - maxoids[:, labels])**2)
            if val < best_val:
                best_val = val
                best_max, best_labels = maxoids, labels
        self.val = best_val
        self.maxoids = best_max
        self.labels = best_labels
        return(best_max, best_labels)

    def _run_once(self, maxit):
        """Runs the algorithm only once and returns the clusters"""

        maxoids = self.Y[:, np.random.choice(
            self.nsamples, size=self.K, replace=False)]
        Y = self.Y.transpose()
        for i in range(maxit):
            # Update Clusters
            D = np.hstack([np.sum((Y-m)**2, axis=1).reshape(self.nsamples, 1)
                          for m in maxoids.transpose()])
            labels = np.argmin(D, axis=1)
            if np.var(labels) == 0:
                break
            # update maxoids
            for k in range(self.K):
                Mk = np.hstack([maxoids[:, 0:k], maxoids[:, k+1:]]).transpose()
                Yk = self.Y[:, labels == k].transpose()
                Mk_br = Mk[np.newaxis, :, :]
                Yk_br = Yk[:, np.newaxis, :]
                summat = np.sum((Yk_br - Mk_br)**2, axis=(1, 2))
                maxoids[:, k] = Yk[np.argmax(summat)]

        return(maxoids, labels)
