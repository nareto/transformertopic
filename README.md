## transformertopic
Topic Modeling using sentence embeddings. The procedure is: 

1. split paragraphs in sentences
2. compute sentence embeddings
3. compute dimension reduction of these embeddings
4. cluster them with HDBSCAN
5. compute a human-readable representation of each cluster/topic

This is inspired by the Topic Modeling procedure described [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6) by Maarten Grootendorst, who also has his own implementation available 
[here](https://github.com/MaartenGr/BERTopic). 

I wanted to code it myself and have features marked with a ⭐, which as far as I know are not available in Grootendorst's implementation.

Features:
- Compute topic modeling 
- Compute dynamic topic modeling ("trends" here)
- ⭐ Assign topics on sentence rather than document level
- ⭐ Experiment with different dimension reducers
- ⭐ Experiment with different ways to generate a wordcloud from a topic
- ⭐ Infer topics of new batches of docs without retraining


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
