import json
from google.colab import drive
import pandas as pd
import spacy
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

#Loading Data 
drive.mount('/content/gdrive')
file_path="/content/gdrive/My Drive/news.article.json"
with open(file_path,"r") as f:
  file=json.load(f)

#Storing the data as dataframe
titles = []
for i in range(len(file)):
  titles.append(file[i]['title'])
date = []
for i in range(len(file)):
  date.append(file[i]['scrapedDate']['$date'])
data = pd.DataFrame({'title':titles,'date':date})
data = data.drop_duplicates(subset='title').reset_index(drop=True)
data = data.dropna()
data.head()

#Preprocessing the data
nlp = spacy.load("en_core_web_lg")
sent_vecs = {}
docs = []
for title in tqdm(data.title):
    doc = nlp(title)
    docs.append(doc)
    sent_vecs.update({title: doc.vector})
sentences = list(sent_vecs.keys())
vectors = list(sent_vecs.values())

#Dividing the data into clusters
x = np.array(vectors)
n_classes ={}
for i in tqdm(np.arange(0.001,0.002)):
  dbscan = DBSCAN(eps=i, min_samples=2, metric='cosine').fit(x)
  n_classes.update({i:len(pd.Series(dbscan.labels_).value_counts())})
dbscan = DBSCAN(eps=0.08, min_samples=2, metric='cosine').fit(x)

#Creating Titles for Clusters
clustered_data = data.copy()
clustered_data['cluster'] = dbscan.labels_
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedded_data = model.encode(clustered_data['title'])
cluster_centroids = {}
for cluster in clustered_data['cluster'].unique():
    cluster_data = embedded_data[clustered_data['cluster'] == cluster]
    centroid = np.mean(cluster_data, axis=0)
    cluster_centroids[cluster] = centroid

def generate_short_title(cluster_centroid, cluster_data):
    similarities = model.similarity(cluster_centroid, embedded_data)
    top_similarities = np.argsort(-similarities)[:3]
    short_title = ' '.join([cluster_data.iloc[i]['title'].values[0] for i in top_similarities])
    return short_title

cluster_titles = {cluster: generate_short_title(centroid, clustered_data) for cluster, centroid in cluster_centroids.items()}
clustered_data['cluster_title'] = clustered_data['cluster'].map(cluster_titles)

#Filtering relevant events
related_event = clustered_data[clustered_data['cluster_title'].str.contains('israel|hamas|gaza|war')]
related_event['date'] = pd.to_datetime(related_event['date'])
related_event.loc[:, 'start_date'] = related_event['date'].dt.strftime('%d %B')
related_event.loc[:, 'end_date'] = related_event['date'].dt.strftime('%d %B')

#Creating and printing timeline
cluster_timelines = related_event.loc[:, ['start_date', 'end_date']].groupby(related_event['cluster_title']).agg({'start_date': 'min', 'end_date': 'max'}).reset_index()
cluster_timelines['timeline'] = cluster_timelines.apply(lambda row: f"{row['start_date']} - {row['end_date']}: {row['cluster_title']}", axis=1)

print(cluster_timelines['timeline'])

