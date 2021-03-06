# %%
%load_ext autoreload
%autoreload 2

from transformertopic.dimensionReducers import PacmapEmbeddings, UmapEmbeddings, TsneEmbeddings
from transformertopic import TransformerTopic
from transformertopic.clusterRepresentators import TextRank, Tfidf, KMaxoids
import pandas as pd

# the following sentences are extracted from some COVID-related articles on https://www.huffingtonpost.co.uk
texts = [
    ["2021-09-30 09:38:47+00:00", "Thousands Of Long Covid Cases In Kids Could Be Averted. Here's How. New research suggests the benefits of children getting two doses of the vaccine outweigh the risks."
     ],
    ["2021-09-29 04:00:04+00:00", "There's A Huge Perk To Getting Vaccinated After Having Covid. People who've had coronavirus and then get the vaccine may be at a marked advantage."],
    ["2021-09-28 04:42:00+00:00",
        "Telegraph's Cartoonist Bob Moran Violates Twitter Rules With Anti-Vax Attack On Doctor."],
    ["2021-09-28 11:26:12+00:00",
        "Here's How Much Face Masks (Still) Reduce Your Risk Of Catching Covid. Just another reason why you should keep wearing your mask this winter."],
    ["2021-04-19 04:47:00+00:00", "India Added To UK's Covid Travel 'Red List', Announces Matt Hancock. The restrictions will come into force from Friday morning."],
    ["2021-04-19 03:52:00+00:00", "Keir Starmer In Pub Clash With Man Spreading Covid 'Misinformation'. Labour leader said he 'profoundly disagreed' with man who angrily opposed lockdown."],
    ["2021-04-19 12:02:16+00:00",
        "My Son Just Turned One. My First Year As A Mum Has Been Lonely, Scary And Confusing."]
]
df = pd.DataFrame(texts, columns=["date", "text"])
df["date"] = pd.to_datetime(df["date"])
df
# %%

reducer = UmapEmbeddings(umapNNeighbors=3)
tt = TransformerTopic(dimensionReducer=reducer, hdbscanMinClusterSize=2)
tt.train(documentsDataFrame=df, dateColumn='date',
         textColumn='text', copyOtherColumns=True)
print(f"Found {tt.nTopics} topics")
# print(tt.df.info())

# %% Show sizes of largest topics
N = 1
topNtopics = tt.showTopicSizes(N)


# %% Choose a cluster representator and show wordclouds for the biggest topics

representator = Tfidf()
# representator = TextRank()
# representator = KMaxoids()
tt.showWordclouds(topNtopics, clusterRepresentator=representator)

#%%
dftp = tt.showTopicTrends(resamplePeriod="1d", normalize=False)
#%%
dftp = tt.showTopicTrends(topicsToShow=topNtopics, resamplePeriod="1d", normalize=True)
#%%
topicsToShow = [0,1]
resamplePeriod = '1M'
date_range = tt.df[tt.df['topic'] != -1].set_index('date')
alltimes = date_range.resample(resamplePeriod).count()['id']

resampledDfs = []
resampledColumns = []
for e, n in enumerate(topicsToShow):
    tseries = tt.df.loc[tt.df["topic"] == n, ["date", "topic"]]
    tseries = tseries.set_index("date")
    resampled = tseries.resample(resamplePeriod).count().rename(
        {"topic": 'count'}, axis=1)['count']
    resampled = resampled.reindex(alltimes.index, method='ffill')
    resampled.sort_index(inplace=True)
    resampledDfs.append(resampled)
    resampledColumns.append('Topic %d' % n)
resampledDfs
# %%
date_range
alltimes
# tt.df
tot = resampledDfs[0].fillna(0) + resampledDfs[1].fillna(0)
tot
resampledDfs[0]/tot
4/6
list(range(tt.nTopics))
#%%
drsf = {tid: resampledDfs[tid] for tid in topicsToShow}
#%%
from functools import reduce
reduce(lambda x,y: x + y, drsf.values())