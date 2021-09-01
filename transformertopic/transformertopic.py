import datetime as dt
import pickle

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English

from transformertopic.clusterRepresentators import TextRank
from transformertopic.dimensionReducers import UmapEmbeddings
from transformertopic.utils import generateTextId, scrambleDateColumn, showWordCloudFromScoresDict

Sentencizer = English()
Sentencizer.add_pipe("sentencizer")


class TransformerTopic():
    """
    Class representing a BertTopic model.
    """

    def __init__(self, dimensionReducer=None, hdbscanMinClusterSize=100, stEmbeddings=None):
        """
        hdbscanMinClustersize: hdbscan parameter. Corresponds to minimum size of topic. Higher => fewer topics.
        """
        if dimensionReducer is None:
            dimensionReducer = UmapEmbeddings()
        self.dimensionReducer = dimensionReducer
        # MODEL PARAMETERS:
        if stEmbeddings is None:
            stEmbeddings = "paraphrase-MiniLM-L6-v2"
        self.stEmbeddingsModel = stEmbeddings
        self.hdbscanMinClusterSize = hdbscanMinClusterSize
        self.hdbscanMetric = 'euclidean'
        self.hdbscanClusterSelectionMethod = 'eom'

        # INPUT DATA
        # self.originalDocumentsDataFrame = None
        # "id" = None
        # "date" = None
        # self.textColumn = None
        self.nOriginalDocuments = 0
        self.df = None
        self.clusterRepresentator = None

        # GENERATED DATA
        # self.topicIds = None
        self.twoDEmbeddings = None
        self.nBatches = 0
        self.nTopics = -1
        self.runFullCompleted = False
        # self.documentIdsToTopics = None  # TODO: can remove? use fulldf instead
        # self.topicsToDocumentIds = None  # TODO: can remove? use fulldf instead
        self.clusterRepresentations = None
        self.topicSizes = None

    def savePickle(self, filepath):
        f = open(filepath, 'wb')
        pickle.dump(self.__dict__, f, protocol=4)
        f.close()
        logger.debug(f"Pickled class to {filepath}")

    def loadPickle(self, filepath):
        f = open(filepath, 'rb')
        tmpdict = pickle.load(f)
        f.close()
        self.__dict__.update(tmpdict)
        logger.debug(f"Loaded class from {filepath}")

    def train(
            self,
            documentsDataFrame,
            dateColumn,
            textColumn,
            idColumn=None,
            copyOtherColumns=False
    ):
        """
        Runs the full clustering procedure - slow.

        documentsDataFrame: dataFrame containing documents
        idColumn: name of column containing unique id of document
        dateColumn: name of column containing date of document
        textColumn: name of column containing text of document
        """
        logger.debug("runFull: start")
        self.nOriginalDocuments = len(documentsDataFrame)
        self.df = documentsDataFrame.copy()
        self.df = pd.DataFrame(self._getSplitSentencesData(
            dataFrame=self.df,
            dateColumn=dateColumn,
            textColumn=textColumn,
            idColumn=idColumn,
            copyOtherColumns=copyOtherColumns
        ))
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df['batch'] = 1
        self.nBatches = 1
        texts = list(self.df["text"])
        # textIds = list(self.df["id"])
        logger.debug(
            f"computing SentenceTransformer embeddings for {self.stEmbeddingsModel}")
        self.stModel = SentenceTransformer(self.stEmbeddingsModel)
        self.stEmbeddings = {}
        self.stEmbeddings[1] = self.stModel.encode(
            texts, show_progress_bar=False)
        self.reducedEmbeddings = {}
        self.reducedEmbeddings[1] = self.dimensionReducer.fit_transform(
            self.stEmbeddings[1])
        logger.debug(
            f"Computing HDBSCAN with min_cluster_size = {self.hdbscanMinClusterSize}, metric = {self.hdbscanMetric}, cluster_selection_method = {self.hdbscanClusterSelectionMethod}"
        )
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=self.hdbscanMinClusterSize,
                                         metric=self.hdbscanMetric,
                                         cluster_selection_method=self.hdbscanClusterSelectionMethod,
                                         prediction_data=True)
        self.clusters = {}
        self.clusters[1] = self.clusterer.fit(self.reducedEmbeddings[1])
        # self.documentIdsToTopics = {}
        # self.topicsToDocumentIds = {k: set() for k in self.clusters[1].labels_}
        for doubleIdx, label in np.ndenumerate(self.clusters[1].labels_):
            idx = doubleIdx[0]
            # tId = textIds[idx]
            # self.documentIdsToTopics[tId] = label
            # self.topicsToDocumentIds[label].add(tId)
            self.df.at[idx, "topic"] = int(label)
        self.nTopics = self.clusters[1].labels_.max() + 1
        self.runFullCompleted = True
        logger.debug("runFull: completed")

    def _getSplitSentencesData(self,
                               dataFrame,
                               dateColumn,
                               textColumn,
                               idColumn=None,
                               copyOtherColumns=False):
        data = []
        for index, row in dataFrame.iterrows():
            date = row[dateColumn]
            fulltext = row[textColumn]
            if idColumn is None:
                id = generateTextId(fulltext)
            else:
                id = row[idColumn]
            if type(fulltext) == type(1.0):
                continue
            sents = Sentencizer(fulltext).sents
            for sent in sents:
                newRow = {
                    "id": id,
                    "date": date,
                    "text": str(sent)
                }
                if copyOtherColumns:
                    for column in set(dataFrame.columns).difference({"id", "date", "text"}):
                        newRow[column] = row[column]
                data.append(newRow)
        return data

    def getTopicsForDoc(self, documentId):
        subdf = self.df.loc[self.df["id"] == documentId]
        return list(subdf["topic"])

    def infer(self, newDocumentsDataFrame, dateColumn, textColumn, idColumn=None):
        """
        Runs HDBSCAN approximate inference on new texts.

        The new DataFrame needs to have the same id, text and date columns as the original one.
        """
        if not self.runFullCompleted:
            raise Exception("No model computed")

        tmpDf = pd.DataFrame(self._getSplitSentencesData(
            dataFrame=newDocumentsDataFrame,
            dateColumn=dateColumn,
            textColumn=textColumn,
            idColumn=idColumn
        ))
        indexesAlreadyPresent = set(
            self.df["id"]).intersection(set(tmpDf["id"]))
        tmpDf = tmpDf[~tmpDf["id"].isin(indexesAlreadyPresent)]
        batch = self.nBatches + 1
        tmpDf['batch'] = batch
        texts = list(tmpDf["text"])
        textIds = list(tmpDf["id"])
        # sentence transformer
        logger.debug(
            f"computing SentenceTransformer embeddings for {self.stEmbeddingsModel}")
        self.stEmbeddings[batch] = self.stModel.encode(
            texts, show_progress_bar=False)
        self.reducedEmbeddings[batch] = self.dimensionReducer.fit_transform(
            self.stEmbeddings[batch])
        logger.debug(
            f"Computing HDBSCAN with min_cluster_size = {self.hdbscanMinClusterSize}, metric = {self.hdbscanMetric}, cluster_selection_method = {self.hdbscanClusterSelectionMethod}"
        )
        # hdbscan inference
        labels, strengths = hdbscan.approximate_predict(
            self.clusterer, self.reducedEmbeddings[batch])
        # assign topics in tmpDf
        for doubleIdx, label in np.ndenumerate(labels):
            idx = doubleIdx[0]
            tId = textIds[idx]
            tmpDf.loc[tmpDf["id"] == tId, "topic"] = int(label)
        self.df = self.df.append(tmpDf)
        self.nBatches += 1
        logger.debug("inference completed")

    def _compute2dEmbeddings(self, batch):
        if not self.runFullCompleted:
            raise Exception("No model computed")
        logger.debug("_compute2dEmbeddings: start")
        if self.twoDEmbeddings is None:
            self.twoDEmbeddings = {}
        self.twoDEmbeddings[batch] = self.dimensionReducer.fit_transform2d(
            self.stEmbeddings[batch])
        logger.debug("_compute2dEmbeddings: completed")

    def plotClusters(self, batch=1):
        if not self.runFullCompleted:
            raise Exception("No model computed")

        if self.twoDEmbeddings is None or batch not in self.twoDEmbeddings.keys():
            self._compute2dEmbeddings(batch)

        logger.debug("plotClusters")

        result = pd.DataFrame(self.twoDEmbeddings[batch], columns=['x', 'y'])
        result['labels'] = self.clusters[batch].labels_

        # Visualize clusters
        fig, ax = plt.subplots(figsize=(20, 10))
        outliers = result.loc[result.labels == -1, :]
        clustered = result.loc[result.labels != -1, :]
        plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
        plt.scatter(clustered.x, clustered.y,
                    c=clustered.labels, s=0.05, cmap='hsv_r')
        plt.colorbar()
        plt.show()

    def _computeClusterRepresentations(self, topics=None, nKeywords=25):
        """
        Computes representation of clusters for wordclouds.
        """
        if not self.runFullCompleted:
            raise Exception("No model computed")
        if topics is None:
            topics = set(range(self.nTopics))
        if self.clusterRepresentations is None:
            self.clusterRepresentations = {k: {} for k in range(-1, self.nTopics)}
        firstbatchdf = self.df[self.df['batch'] == 1]
        print(f"Computing cluster representations for topics {topics}")
        for cluster_idx in tqdm(topics):
            topicDf = firstbatchdf[firstbatchdf['topic'] == cluster_idx]
            documents = list(topicDf["text"])
            keywords, scores = self.clusterRepresentator.fit_transform(
                documents, nKeywords)
            assert len(keywords) == len(scores)
            self.clusterRepresentations[cluster_idx] = {
                keywords[i]: scores[i] for i in range(len(keywords))}

    def showWordclouds(self, topicsToShow=None, nWordsToShow=15, clusterRepresentator=None):
        """
        Show wordclouds.

        topicsToShow: set with topics indexes to print. If None all topics are chosen.
        nWordsToShow: how many words to show for each topic
        """
        if clusterRepresentator is None and self.clusterRepresentator is None:
            raise Exception("You need to pass a clusterRepresentator")
        topicsToShowSet = set(topicsToShow)
        if topicsToShow is None:
            topicsToShow = list(range(self.nTopics))
        else:
            if set(range(self.nTopics)).intersection(topicsToShowSet) != topicsToShowSet:
                raise Exception(f"topicsToShow cannot include topics outside of -1..{self.nTopics - 1}")
        if type(clusterRepresentator) != type(self.clusterRepresentator):
            self.clusterRepresentator = clusterRepresentator
            topicsToCompute = topicsToShowSet
        else:
            assert self.clusterRepresentations is not None
            topicsToCompute = topicsToShowSet.difference(set(self.clusterRepresentations.keys()))
        if len(topicsToCompute) > 0:
            self._computeClusterRepresentations(topicsToCompute)
        for topicIdx in topicsToShow:
            print("Topic %d" % topicIdx)
            wordScores = self.clusterRepresentations[topicIdx]
            showWordCloudFromScoresDict(wordScores)

    def prettyPrintTopics(self, topicsToPrint=None, nWordsToShow=5):
        """
        Pretty prints topics.

        topicsToPrint: set with topics indexes to print. If None all topics are chosen.
        nWordsToShow: how many words to show for each topic
        """
        if self.clusterRepresentations is None:
            self._computeClusterRepresentations()
        if topicsToPrint is None:
            topicsToPrint = set(range(self.nTopics))

        for topicIdx in topicsToPrint:
            print(f"\nTopic {topicIdx}")
            representationItems = self.clusterRepresentations[topicIdx].items()
            wordFrequencies = sorted(
                representationItems, key=lambda x: x[1], reverse=True)
            for word, frequency in wordFrequencies[:nWordsToShow]:
                print("\n%10s:%2.3f" % (word, frequency))

    def showTopicSizes(self, showTopNTopics=None, minSize=None, showNoTopic=False, batches=None):
        """
        Show bar chart with topic sizes (n. of documents).

        Returns list of topic indexes with more than minSize documents.

        showTopNTopics: if integer, show only this number of largest sized topics. If None it is ignored. 
        minSize: show only topics with bigger size than this. If None ignore
        showNoTopic: show also topic -1, i.e. "noise" documents.
        batches: which batches to include. If None include all known documents.
        """

        if batches is None:
            batches = {k for k in range(1, self.nBatches+1)}
        if minSize is None and showTopNTopics is None:
            plotIndexIsRange = True
        else:
            plotIndexIsRange = False
        topicIndexes = []
        topicSizes = []
        df = self.df
        for k in range(self.nTopics):
            # docsK = self.topicsToDocumentIds[k]
            docsK = df.loc[(df['batch'].isin(batches)) & (df["topic"] == k)]
            ndocs = len(docsK)
            if minSize is None or ndocs > minSize:
                if plotIndexIsRange:
                    topicIndexes.append(k)
                else:
                    topicIndexes.append(str(k))
                topicSizes.append(ndocs)
        # logger.debug(f"batches: {batches}, topicsizes: {topicSizes}")
        if showTopNTopics is None:
            indexes = topicIndexes
            sizes = topicSizes
        else:
            argsort = np.argsort(topicSizes)[::-1]
            indexes = []
            sizes = []
            for i in argsort[:showTopNTopics]:
                # indexes = topicIndexes[:showTopNTopics]
                if plotIndexIsRange:
                    indexes.append(topicIndexes[i])
                else:
                    indexes.append(str(topicIndexes[i]))
                sizes.append(topicSizes[i])
        # print(f"index: {indexes}, sizes: {sizes}")
        plt.bar(indexes, sizes)
        plt.show()

        return [int(k) for k in indexes]

    def showTopicTrends(self,
                        topicsToShow=None,
                        batches=None,
                        resamplePeriod='6M',
                        scrambleDates=False,
                        normalize=False
                        ):
        """
        Show a time plot of popularity of topics.

        topicsToShow: set with topics indexes to print. If None all topics are chosen.
        batches: which batches to include. If None include all known documents.
        resamplePeriod: resample to pass to pandas.DataFrame.resample.
        scrambleDates: if True, redistributes dates that are on 1st of the month (year) uniformly in that month (year)
        """

        if topicsToShow is None:
            topicsToShow = range(self.nTopics)
        if batches is None:
            batches = {k for k in range(1, self.nBatches+1)}
        if(scrambleDates):
            df = scrambleDateColumn(self.df, "date")
        else:
            df = self.df

        date_range = self.df[self.df['topic'] != -1].set_index('date')
        alltimes = date_range.resample(resamplePeriod).count()['id']
        df = df[df['batch'].isin(batches)]
        resampledDfs = []
        resampledColumns = []
        for e, n in enumerate(topicsToShow):
            tseries = df.loc[df["topic"] == n, ["date", "topic"]]
            tseries = tseries.set_index("date")
            resampled = tseries.resample(resamplePeriod).count().rename(
                {"topic": 'count'}, axis=1)['count']
            resampled = resampled.reindex(alltimes.index, method='ffill')
            resampled.sort_index(inplace=True)
            resampledDfs.append(resampled)
            resampledColumns.append('Topic %d' % n)
        if normalize:
            totsum = resampledDfs[0].fillna(0)
            for rs in resampledDfs[1:]:
                totsum = totsum + rs.fillna(0)
            normalizedDfs = []
            for rs in resampledDfs:
                normalizedDfs.append(rs/totsum)
            resampledDfs = normalizedDfs
        dfToPlot = pd.DataFrame(columns=resampledColumns, index=alltimes.index)
        for e, rdf in enumerate(resampledDfs):
            topicColumn = resampledColumns[e]
            dfToPlot[topicColumn] = rdf
        dfToPlot.interpolate(method='linear').plot()
