#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple

class Tfidf():
    def __init__(self) -> None:
        pass

    def fit_transform(self, documents, nKeywords=10) -> Tuple[List[float], List[str]]:
        text = (". ".join(documents),)
        tfidf = TfidfVectorizer(stop_words="english").fit(text)
        X = tfidf.transform(text).toarray()
        argsort = X.argsort()[0,::-1][:nKeywords]
        scores = X[0,argsort]
        keywords = [tfidf.get_feature_names()[i] for i in argsort]
        return (keywords, scores)


#%%
