import datetime as dt
from logging import raiseExceptions
import pickle

import hdbscan
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English

from transformertopic.clusterRepresentators import TextRank
from transformertopic.clusterRepresentators.tfidf import Tfidf
from transformertopic.dimensionReducers import UmapEmbeddings
from transformertopic.utils import generateTextId, scrambleDateColumn, showWordCloudFromScoresDict

Sentencizer = English()
Sentencizer.add_pipe("sentencizer")


class TransformerTopic():
    """
    Class representing a BertTopic model.
    """

    def __init__(self, dimensionReducer=None, hdbscanMinClusterSize=25, stEmbeddings=None):
        """
        hdbscanMinClustersize: hdbscan parameter. Corresponds to minimum size of topic. Higher => fewer topics.
        stEmbeddings: embeddings model for SentenceTransformer. See available models at https://huggingface.co/sentence-transformers?sort=downloads
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
        self.nOriginalDocuments = 0
        self.df = None
        self.clusterRepresentator = None

        # GENERATED DATA
        self.twoDEmbeddings = None
        self.nBatches = 0
        self.nTopics = -1
        self.runFullCompleted = False
        self.clusterRepresentations = None
        self.topicSizes = None
        self.stEmbeddings = None
        self.reducedEmbeddings = None
        self.clusters = None
        self.topicNames = None

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

    def saveCsv(self, filepath):
        self.df.to_csv(filepath)

    def loadCsv(self, filepath, dateColumn='date', textColumn='text'):
        self.df = pd.read_csv(filepath)
        self.df.rename(columns={dateColumn: 'date', textColumn: 'text'})
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.nTopics = 1 + int(self.df['topic'].max())
        self.nBatches = int(self.df['batch'].max())

    def train(
            self,
            documentsDataFrame=None,
            dateColumn='date',
            textColumn='text',
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
        logger.debug("train: start")
        if documentsDataFrame is not None:
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
        elif self.df is None:
            raise(
                Exception("Either pass documentsDataFrame or self.df should not be None"))
        texts = list(self.df["text"])
        # textIds = list(self.df["id"])
        if self.stEmbeddings is None:
            logger.debug(
                f"train: computing SentenceTransformer embeddings for {self.stEmbeddingsModel}")
            self.stModel = SentenceTransformer(self.stEmbeddingsModel)
            self.stEmbeddings = {}
            self.stEmbeddings[1] = self.stModel.encode(
                texts, show_progress_bar=False)
        if self.reducedEmbeddings is None:
            self.reducedEmbeddings = {}
            self.reducedEmbeddings[1] = self.dimensionReducer.fit_transform(
                self.stEmbeddings[1])
        if self.clusters is None:
            logger.debug(
                f"train: computing HDBSCAN with min_cluster_size = {self.hdbscanMinClusterSize}, metric = {self.hdbscanMetric}, cluster_selection_method = {self.hdbscanClusterSelectionMethod}"
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
        logger.debug("train: completed")

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
            f"infer: computing SentenceTransformer embeddings for {self.stEmbeddingsModel}")
        self.stEmbeddings[batch] = self.stModel.encode(
            texts, show_progress_bar=False)
        self.reducedEmbeddings[batch] = self.dimensionReducer.fit_transform(
            self.stEmbeddings[batch])
        logger.debug(
            f"infer: computing HDBSCAN with min_cluster_size = {self.hdbscanMinClusterSize}, metric = {self.hdbscanMetric}, cluster_selection_method = {self.hdbscanClusterSelectionMethod}"
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
        logger.debug("infer: inference completed")

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

    def _computeClusterRepresentations(self,
                                       topics=None,
                                       nKeywords=25,
                                       clusterRepresentator=None,
                                       ):
        """
        Computes representation of clusters for wordclouds.
        """
        if topics is None:
            topics = range(self.nTopics)
        topicSet = set(topics)
        if clusterRepresentator is None and self.clusterRepresentator is None:
            self.clusterRepresentator = Tfidf()
        if self.clusterRepresentator is None or (clusterRepresentator is not None and clusterRepresentator != self.clusterRepresentator):
            self.clusterRepresentator = clusterRepresentator
            self.clusterRepresentations = {}
            self.topicNames = {}
            topicsToCompute = topicSet
        else:
            assert self.clusterRepresentations is not None
            topicsToCompute = topicSet.difference(
                set(self.clusterRepresentations.keys()))
        firstbatchdf = self.df[self.df['batch'] == 1]
        if len(topicsToCompute) == 0:
            return
        print(
            f"Computing cluster representations for topics {topicsToCompute}")
        for cluster_idx in tqdm(topicsToCompute):
            topicDf = firstbatchdf[firstbatchdf['topic'] == cluster_idx]
            documents = list(topicDf["text"])
            keywords, scores = self.clusterRepresentator.fit_transform(
                documents, nKeywords)
            assert len(keywords) == len(scores)
            self.clusterRepresentations[cluster_idx] = {
                keywords[i]: scores[i] for i in range(len(keywords))}
            self.topicNames[cluster_idx] = "_".join(
                keywords[:4])+"."+str(cluster_idx)

    def showWordclouds(self,
                       topicsToShow=None,
                       nWordsToShow=25,
                       clusterRepresentator=None,
                       saveToFilepath=None
                       ):
        """
        Computes cluster representations and uses them to show wordclouds.

        topicsToShow: set with topics indexes to print. If None all topics are chosen.
        nWordsToShow: how many words to show for each topic
        clusterRepresentator: an instance of a clusterRepresentator 
        saveToFilepath: save the wordcloud to this filepath
        """
        self._computeClusterRepresentations(
            topics=topicsToShow,
            nKeywords=nWordsToShow,
            clusterRepresentator=clusterRepresentator
        )
        for topicIdx in topicsToShow:
            print("Topic %d" % topicIdx)
            wordScores = self.clusterRepresentations[topicIdx]
            showWordCloudFromScoresDict(
                wordScores, max_words=nWordsToShow, filepath=saveToFilepath)

    def searchForWordInTopics(self,
                              word,
                              topicsToSearch=None,
                              topNWords=15,
                              clusterRepresentator=None):
        """
        Returns any topic containing 'word' in the topNWords of its cluster representation

        word: the word to look for
        topicsToSearch: set with topics indexes to search in. If None all topics are searched in.
        nWordsToShow: how many words to show for each topic
        clusterRepresentator: an instance of a clusterRepresentator 
        """
        if topicsToSearch is None:
            topicsToSearch = set(range(self.nTopics))
        self._computeClusterRepresentations(
            topics=topicsToSearch,
            nKeywords=topNWords,
            clusterRepresentator=clusterRepresentator
        )
        positive_topics = set()
        for tidx in topicsToSearch:
            if word in self.clusterRepresentations[tidx].keys():
                positive_topics.add(tidx)
        return positive_topics

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

    def showTopicSizes(self,
                       showTopNTopics=None,
                       minSize=None,
                       batches=None):
        """
        Show bar chart with topic sizes (n. of documents).

        Returns list of topic indexes with more than minSize documents.

        showTopNTopics: if integer, show only this number of largest sized topics. If None it is ignored. 
        minSize: show only topics with bigger size than this. If None ignore
        batches: which batches to include. If None include all known documents.
        """

        if batches is None:
            batches = {k for k in range(1, self.nBatches+1)}
        if minSize is None and showTopNTopics is None:
            plotIndexIsRange = True
        else:
            plotIndexIsRange = False
        topicIndexes = []
        self.topicSizes = []
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
                self.topicSizes.append(ndocs)
        # logger.debug(f"batches: {batches}, self.topicSizes: {self.topicSizes}")
        if showTopNTopics is None:
            indexes = topicIndexes
            sizes = self.topicSizes
        else:
            argsort = np.argsort(self.topicSizes)[::-1]
            indexes = []
            sizes = []
            for i in argsort[:showTopNTopics]:
                # indexes = topicIndexes[:showTopNTopics]
                if plotIndexIsRange:
                    indexes.append(topicIndexes[i])
                else:
                    indexes.append(str(topicIndexes[i]))
                sizes.append(self.topicSizes[i])
        # print(f"index: {indexes}, sizes: {sizes}")
        # plt.bar(indexes, sizes)
        # plt.show()
        plot_ = sns.barplot(x=indexes, y=sizes, palette='colorblind')
        if len(indexes) > 25:
            modulo = len(indexes) // 20
            for ind, label in enumerate(plot_.get_xticklabels()):
                if ind % modulo == 0:  # every 10th label is kept
                    label.set_visible(True)
                else:
                    label.set_visible(False)
        return [int(k) for k in indexes]

    def showTopicTrends(self,
                        topicsToShow=None,
                        batches=None,
                        resamplePeriod='6M',
                        scrambleDates=False,
                        normalize=False,
                        fromDate=None,
                        toDate=None,
                        saveToFilepath=None,
                        **plotkwargs
                        ):
        """
        Show a time plot of popularity of topics. On the y-axis the count of sentences in that topic is shown. If normalize is set to True, the percentage of sentences in that topic (when considering all the sentences in the whole corpus in that time slot) is shown.

        topicsToShow: set with topics indexes to print. If None all topics are chosen.
        batches: which batches to include. If None include all known documents.
        resamplePeriod: resample to pass to pandas.DataFrame.resample.
        normalize: if False count of sentences is shown. If True percentages relative to the whole corpus.
        scrambleDates: if True, redistributes dates that are on 1st of the month (year) uniformly in that month (year)
        """

        self._computeClusterRepresentations(topics=topicsToShow)
        if topicsToShow is None:
            topicsToShow = range(self.nTopics)
        if batches is None:
            batches = {k for k in range(1, self.nBatches+1)}
        if(scrambleDates):
            df = scrambleDateColumn(self.df, "date")
        else:
            df = self.df

        # we need to build a common index to plot all the resampled time series against
        date_range = self.df[self.df['topic'] != -1].set_index('date')
        alltimes = date_range.resample(resamplePeriod).count()['id']

        df = df[df['batch'].isin(batches)]
        resampledDfs = {}
        resampledColumns = {}
        topicsToResample = topicsToShow if not normalize else list(
            range(self.nTopics))
        for topicIdx in topicsToResample:
            tseries = df.loc[df["topic"] == topicIdx, ["date", "topic"]]
            tseries = tseries.set_index("date")
            resampled = tseries.resample(resamplePeriod).count().rename(
                {"topic": 'count'}, axis=1)['count']
            # use the common index
            resampled = resampled.reindex(alltimes.index, method='ffill')
            resampled.sort_index(inplace=True)
            resampledDfs[topicIdx] = resampled.fillna(0)
            resampledColumns[topicIdx] = self.topicNames[topicIdx] if topicIdx in self.topicNames else 'Topic %d' % topicIdx
        if normalize:
            from functools import reduce
            totsum = reduce(lambda x, y: x + y, resampledDfs.values())
            normalizedDfs = {}
            for topicIdx, rs in resampledDfs.items():
                normalizedDfs[topicIdx] = (rs/totsum).fillna(0)
            resampledDfs = normalizedDfs

        # dfToPlot = pd.DataFrame(columns=[resampledColumns[tidx] for tidx in topicsToShow], index=alltimes.index)
        dfToPlot = pd.DataFrame(index=alltimes.index)
        for topicIdx in topicsToShow:
            rsDf = resampledDfs[topicIdx]
            column = resampledColumns[topicIdx]
            dfToPlot[column] = rsDf
        if fromDate is not None:
            dfToPlot = dfToPlot.loc[dfToPlot.index > fromDate]
        if toDate is not None:
            dfToPlot = dfToPlot.loc[dfToPlot.index < toDate]
        axis = dfToPlot.interpolate(method='linear').plot(**plotkwargs)
        if saveToFilepath is not None:
            axis.figure.savefig(saveToFilepath)
        return dfToPlot
