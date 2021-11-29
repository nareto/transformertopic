# transformertopic
Topic Modeling using sentence embeddings. This procedure works very well: in practice it almost always produces sensible topics and (from a practical point of view) renders all LDA variants obsolete. 

This is my own implementation of the procedure described [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6) by *Maarten Grootendorst*, who also has his own implementation available 
[here](https://github.com/MaartenGr/BERTopic). Thanks for this brilliant idea! 

I wanted to code it myself and have features marked with a ⭐, which as far as I know are not available in Grootendorst's implementation.

Features:
- Compute topic modeling 
- Compute dynamic topic modeling ("trends" here)
- ⭐ Assign topics on sentence rather than document level
- ⭐ Experiment with different dimension reducers
- ⭐ Experiment with different ways to generate a wordcloud from a topic
- ⭐ Infer topics of new batches of docs without retraining

# How it works
In the following the words "cluster" and "topic" are used interchangeably. Please note that in classic Topic Modeling procedures (e.g. those based on [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)) each document is a probability distribution over topics. In this sense the procedure here presented could be considered as a special case where these distributions are always degenerate and concentrate the probability on one single index.

The procedure is: 

1. split paragraphs into sentences
2. compute sentence embeddings (using [sentence transformers](https://github.com/UKPLab/sentence-transformers))
3. compute dimension reduction of these embeddings (with [umap](https://github.com/lmcinnes/umap), [pacmap](https://github.com/YingfanWang/PaCMAP), [tsne](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) or [pca](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html))
4. cluster them with [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) 
5. for each topic compute a "cluster representator": a dictionary with words as keys and ranks as values (using [tfidf](https://en.wikipedia.org/wiki/Tf-idf), [textrank](https://derwen.ai/docs/ptr/) or [kmaxoids](http://ceur-ws.org/Vol-1458/E19_CRC4_Bauckhage.pdf) [^1])
6. use the cluster representators to compute [wordcloud](https://github.com/amueller/word_cloud)s for each topic

[^1]: my own implementation, see [kmaxoids.py](https://github.com/nareto/transformertopic/blob/master/transformertopic/clusterRepresentators/kmaxoids.py)

# Installation

    pip install -U transformertopic

# Usage
View also `test.py`.

Choose a reducer

    from transformertopic.dimensionReducers import PacmapEmbeddings, UmapEmbeddings, TsneEmbeddings
    #reducer = PacmapEmbeddings()
    #reducer = TsneEmbeddings()
    reducer = UmapEmbeddings(umapNNeighbors=13)

Init and run the model

    from transformertopic import TransformerTopic
    tt = TransformerTopic(dimensionReducer=reducer, hdbscanMinClusterSize=20)
    tt.train(documentsDataFrame=pandasDf, dateColumn='date', textColumn='coref_text', copyOtherColumns = True)
    print(f"Found {tt.nTopics} topics")
    print(tt.df.info())

If you want to use different embeddings, you can pass the SentenceTransformer model name via the `stEmbeddings` init argument to `TransformerTopic`. 

Show sizes of largest topics

    N = 10
    topNtopics = tt.showTopicSizes(N)


Choose a cluster representator and show wordclouds for the biggest topics

    from transformertopic.clusterRepresentators import TextRank, Tfidf, KMaxoids
    representator = Tfidf()
    # representator = TextRank()
    tt.showWordclouds(topNtopics clusterRepresentator=representator)

Show frequency of topics over times (dynamic topic modeling), or trends:

    tt.showTopicTrends()

Show topics in which "car" appears in the top 75 words in their cluster representation:

    tt.searchForWordInTopics("car", topNWords=75)