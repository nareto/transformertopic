## transformertopic
Topic Modeling using sentence embeddings. The procedure is: 

1. compute sentence embeddings
2. compute dimension reduction of these
3. cluster them 
4. compute a human-readable representation of each cluster/topic

This is inspired by the Topic Modeling procedure described [here](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6) by Maarten Grootendorst, who also has his own implementation available 
[here](https://github.com/MaartenGr/BERTopic).

# Usage
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

Show sizes of largest topics

    N = 10
    topNtopics = tt.showTopicSizes(N)


Choose a cluster representator and show wordclouds for the biggest topics

    from transformertopic.clusterRepresentators import TextRank, Tfidf, KMaxoids
    representator = Tfidf()
    # representator = TextRank()
    tt.showWordclouds(topNtopics clusterRepresentator=representator)