from flask import Flask, jsonify, request, render_template
from pinecone_service import PineconeService
from ocr_service import OCRService
from web_scrapper_service import WebScraperService
from cnn_service import CNNService
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
import waitress


app = Flask(__name__,template_folder='templates')

model = load_model('cnn_model.h5')

product_data = pd.read_csv('cleaned_dataset.csv')

# Initialize services with your API key and other necessary parameters
pinecone_service = PineconeService(
    api_key='21f2127c-159b-4a3b-9fbc-a9b3a1ec422c',
    index_name='product-vectors',
    dataset_filename='cleaned_dataset.csv'
)
ocr_service = OCRService()
web_scraper_service = WebScraperService()
cnn_service = CNNService('cnn_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

# Tokenizer for text preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(product_data['Description'].values)

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as per your model's requirement
    return padded_sequences

@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    query = request.json.get('query', '')
    
    # Preprocess the query
    preprocessed_query = preprocess_text(query)
    
    # Predict using the model
    predictions = model.predict(preprocessed_query)
    
    # Process the model output to find top matching products
    # This part will depend on your specific model's output format
    top_indices = np.argsort(predictions[0])[::-1][:5]  # Adjust the slicing as needed
    
    # Extract the top matching products
    matching_products = product_data.iloc[top_indices]
    
    products = matching_products.to_dict(orient='records')
    response = "Products matching your query"
    
    return jsonify({"products": products, "response": response})

@app.route('/text-query')
def text_query():
    return render_template('text_query.html')
@app.route('/image-query', methods=['GET', 'POST'])
def image_query():
    if request.method == 'POST':
        image_file = request.files.get('image_data')
        extracted_text = "Sample extracted text"  # Replace with actual OCR processing code
        matched_products = products_df[products_df['Description'].str.contains(extracted_text, case=False, na=False)]
        products = matched_products.to_dict(orient='records')
        response = "Here are the matching products based on your handwritten query."
        return render_template('image_query.html', products=products, response=response)
    return render_template('image_query.html')

@app.route('/product-image-upload', methods=['GET', 'POST'])
def product_image_upload():
    if request.method == 'POST':
        product_image = request.files.get('product_image')
        detected_product = "Sample detected product"  # Replace with actual image processing code
        matched_products = products_df[products_df['Description'].str.contains(detected_product, case=False, na=False)]
        products = matched_products.to_dict(orient='records')
        response = "Here are the matching products based on your product image."
        return render_template('product_image_upload.html', products=products, response=response)
    return render_template('image_upload.html')

@app.route('/sample_response', methods=['GET'])
def sample_response():
    return render_template('index.html')


# Endpoint for product recommendations
@app.route('/recommend', methods=['POST'])
def recommend_products():
    data = request.json
    query_description = data['query']
    product_matches = pinecone_service.recommend_products(query_description)
    return jsonify({
        'products': product_matches,
        'response': f"Here are the top products matching your query '{query_description}'"
    })

# Endpoint for OCR functionality (for handwritten text)
@app.route('/ocr-handwritten', methods=['POST'])
def ocr_handwritten_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    extracted_text = ocr_service.extract_text(image_file)
    product_matches = pinecone_service.recommend_products(extracted_text)
    return jsonify({
        'extracted_text': extracted_text,
        'products': product_matches,
        'response': f"Here are the top products matching your handwritten query"
    })

# Endpoint for web scraping product images
@app.route('/scrape-images', methods=['POST'])
def scrape_images():
    data = request.json
    url = data['url']
    image_urls = web_scraper_service.scrape_images(url)
    return jsonify({
        'image_urls': image_urls,
        'response': f"Scraped {len(image_urls)} images from the URL '{url}'"
    })

# Endpoint for image-based product detection
@app.route('/detect-product', methods=['POST'])
def detect_product():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    class_name = cnn_service.predict_product_class(image_file)
    product_matches = pinecone_service.recommend_products(class_name)
    return jsonify({
        'class_name': class_name,
        'products': product_matches,
        'response': f"Detected product class '{class_name}' and here are the top matching products"
    })

if __name__ == '__main__':
    waitress.serve(app, host='0.0.0.0', port=5000)
