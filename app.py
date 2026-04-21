from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data/restaurant.csv")

df['cuisines'] = df['cuisines'].fillna('')
df['name'] = df['name'].fillna('')

# ML model
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['cuisines'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

df = df.reset_index()

def get_recommendations(name):
    if name not in df['name'].values:
        return None

    idx = df[df['name'] == name].index[0]

    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:7]

    indices = [i[0] for i in scores]

    return df.iloc[indices][['name','cuisines','cost','Mean Rating']].to_dict(orient='records')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search')
def search():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    name = request.form['restaurant']

    results = get_recommendations(name)

    if results is None:
        return "Restaurant not found 😢"

    return render_template('result.html', restaurants=results)


if __name__ == '__main__':
    app.run(debug=True)