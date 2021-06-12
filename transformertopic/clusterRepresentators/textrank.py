#%%
import spacy
import pytextrank
from typing import List, Tuple
import numpy as np

class TextRank():
    def __init__(self) -> None:
        pass

    def fit_transform(self, documents, nKeywords = 10) -> Tuple[List[str], List[np.array]]:
        text = ". ".join(documents)
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
        doc = nlp(text)
        keywords = [p.text for p in doc._.phrases[:nKeywords]]
        scores = np.array([p.rank for p in doc._.phrases[:nKeywords]])
        return (keywords, scores)
#%%

