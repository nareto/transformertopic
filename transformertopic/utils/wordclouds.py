import matplotlib.pyplot as plt
from wordcloud import WordCloud


def showWordCloudFromScoresDict(wordScoreDict, max_words=25):
    """
    Show word-cloud.

    wordFrequencyDict: a dictionary of the form {"word": score}. 
    words with higher scores will be bigger.
    max_words: how many words at maximum to show in the word-cloud.
    """
    wc = WordCloud(background_color="white", max_words=max_words)
    wc.generate_from_frequencies(wordScoreDict)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def showWordCloudFromDocuments(documents, clusterRepresentator, max_words=25):
    """
    Show word-cloud for list of documents using clusterRepresentator.

    documents: iterable of strings of
    clusterRepresentator: an instance of class from clusterRepresentetators
    """
    keywords, scores = clusterRepresentator.fit_transform(
        documents, max_words)
    assert len(keywords) == len(scores)
    wordScoreDict = {keywords[i]: scores[i] for i in range(len(keywords))}
    showWordCloudFromScoresDict(wordScoreDict, max_words)
