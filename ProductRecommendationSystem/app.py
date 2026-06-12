import os
import pandas as pd
import random
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)

trending_products = pd.read_csv('models/trending_products.csv')
train_data = pd.read_csv('models/clean_data.csv')

def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text

def content_based_recommendations(train_data, item_name, top_n=10):
    # Case-insensitive match using substring search
    matches = train_data[train_data['Name'].str.lower().str.contains(item_name.lower(), na=False)]

    if matches.empty:
        print(f"No match found for: {item_name}")
        return pd.DataFrame()

    # Take the first matching item's index
    item_index = matches.index[0]
    print(f"Matched item: {train_data.loc[item_index, 'Name']}")  # Debug print

    # TF-IDF and cosine similarity
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]

    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details

random_image_urls = [
    "static/img_1.png",
    "static/img_2.png",
    "static/img_3.png",
    "static/img_4.png",
    "static/img_5.png",
    "static/img_6.png",
    "static/img_7.png",
    "static/img_8.png",
]

@app.route("/")
def index():
    prices = [40, 50, 150, 70, 100, 200, 106, 100]
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    return render_template('index.html', trending_products=trending_products.head(8),
                           truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(prices))

@app.route("/main")
def main():
    return render_template('main.html', content_based_rec=pd.DataFrame(), load=True)

@app.route("/index")
def indexredirect():
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8),
                           truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = request.form.get('nbr')

        if not prod or not nbr or not nbr.isdigit():
            message = "Please enter a product name and a valid number of products."
            return render_template('main.html',
                                   message=message,
                                   content_based_rec=pd.DataFrame(),
                                   load=False)

        nbr = int(nbr)
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html',
                                   message=message,
                                   content_based_rec=pd.DataFrame(),
                                   load=False)

        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

        return render_template('main.html',
                               content_based_rec=content_based_rec,
                               truncate=truncate,
                               random_product_image_urls=random_product_image_urls,
                               random_price=random.choice(price),
                               load=False)

if __name__ == '__main__':
    app.run(debug=True)
