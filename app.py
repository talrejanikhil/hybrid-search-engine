# Import necessary libraries:
import pandas as pd
from PIL import Image
from flask import Flask, request, render_template, jsonify

from query import search_products

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# Define the route for handling search requests, allowing only POST methods to ensure data privacy and integrity.
@app.route('/search', methods=['POST'])
def search():
    # Extract the search query and other parameters from the form data.
    query = request.form.get('query', '')
    alpha = request.form.get('alpha', 0.0)
    search_query_image = None
    # Check if an image file is part of the request for image-based search functionality.
    if 'image' in request.files:
        search_query_image = Image.open(request.files['image'])
    # Print the form data to the console for debugging purposes.
    print(request.form)
    # Collect additional filter parameters from the form data to apply in the product search.
    query_filter = {
        'gender': request.form.get('gender', ''),
        'baseColour': request.form.get('baseColour', ''),
        'season': request.form.get('season', ''),
        'usage': request.form.get('usage', ''),
    }
    # Perform the product search using the provided query, image input, alpha value, and filters.
    search_results = search_products(query=query, image_input=search_query_image, alpha=float(alpha),
                                     query_filter=query_filter)
    # Extract metadata from the search results to be used in the response.
    meta_data = [s['metadata'] for s in search_results]
    # Add image URLs to the metadata for each record, assuming a naming convention based on the record's ID.
    for record in meta_data:
        record['imageUrl'] = f'image_{str(int(record["id"]))}.jpg'

    # Generate facet options for filtering based on the search results.
    facet_options = get_facets_from_search(meta_data)

    # Return the search response as JSON, including the search results, available facet options, and applied filters.
    return jsonify({'response': meta_data, 'facets': facet_options, 'appliedFilters': query_filter})


# Define a function to extract and organize facet options from the search results metadata.
def get_facets_from_search(meta_data):
    # Convert the list of metadata dictionaries to a pandas DataFrame for easy manipulation.
    df = pd.DataFrame(meta_data)
    # Extract unique values and sort them for each facet
    facet_options = {
        'gender': sorted(list(df['gender'].unique())),
        'baseColour': sorted(list(df['baseColour'].unique())),
        'season': sorted(list(df['season'].unique())),
        'usage': sorted(list(df['usage'].unique())),
    }
    return facet_options


if __name__ == '__main__':
    app.run(
        debug=True)
