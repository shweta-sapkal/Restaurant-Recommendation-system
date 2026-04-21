import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/restaurant.csv")

df['cuisines'] = df['cuisines'].fillna('')
df['name'] = df['name'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['cuisines'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

df = df.reset_index()

def get_recommendations(name):
    if name not in df['name'].values:
        return None

    idx = df[df['name'] == name].index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    indices = [i[0] for i in scores]

    return df.iloc[indices][['name','cuisines','cost','Mean Rating']]