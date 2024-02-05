import pandas as pd
from flask import Flask, request, render_template, jsonify

from query import search_products

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    alpha = request.form.get('alpha', 0.0)
    search_query_image  = None
    if 'image' in request.files:
        search_query_image = request.files['image']
    print(request.form)
    query_filter = {
        'gender': request.form.get('gender', ''),
        'baseColour': request.form.get('baseColour', ''),
        'season': request.form.get('season', ''),
        'usage': request.form.get('usage', ''),
    }
    search_results = search_products(query=query, image_input=search_query_image, alpha=float(alpha),
                                     query_filter=query_filter)
    meta_data = [s['metadata'] for s in search_results]
    # add images
    for record in meta_data:
        record['imageUrl'] = f'image_{str(int(record["id"]))}.jpg'

    facet_options = get_facets_from_search(meta_data)

    return jsonify({'response': meta_data, 'facets': facet_options, 'appliedFilters': query_filter})


def get_facets_from_search(meta_data):
    df = pd.DataFrame(meta_data)
    facet_options = {
        'gender': sorted(list(df['gender'].unique())),
        'baseColour': sorted(list(df['baseColour'].unique())),
        'season': sorted(list(df['season'].unique())),
        'usage': sorted(list(df['usage'].unique())),
    }
    return facet_options


if __name__ == '__main__':
    app.run(debug=True)
