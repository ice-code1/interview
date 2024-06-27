from flask import Flask, jsonify, request, render_template
from pinecone_service import PineconeService
from ocr_service import OCRService
from web_scrapper_service import WebScraperService
from tensorflow.keras.models import load_model
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import MeanSquaredError
from keras.src.legacy.saving import legacy_h5_format
from sklearn.preprocessing import LabelEncoder
import easyocr
import numpy as np
import tensorflow as tf
import waitress
import numpy as np
from tensorflow.keras.models import load_model
import os
import json
from io import BytesIO

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

model = legacy_h5_format.load_model_from_hdf5("text_model1.h5", custom_objects={'mse': 'mse'})

product_data = pd.read_csv('cleaned_dataset.csv')

# Initialize services with your API key and other necessary parameters
pinecone_service = PineconeService(
    api_key='21f2127c-159b-4a3b-9fbc-a9b3a1ec422c',
    index_name='product-vectors',
    dataset_filename='cleaned_dataset.csv'
)
ocr_service = OCRService()
web_scraper_service = WebScraperService()
#cnn_service = CNNService('cnn_model.h5')

# Initialize the tokenizer and fit on product descriptions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(product_data['Description'].values)

stock_code_encoder = LabelEncoder()
product_data['StockCodeEncoded'] = stock_code_encoder.fit_transform(product_data['StockCode'])

country_encoder = LabelEncoder()
product_data['CountryEncoded'] = country_encoder.fit_transform(product_data['Country'])

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    return padded_sequences


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/text-query')
def text_query():
    return render_template('text_query.html')

@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    products = []
    response = "No input provided."

    if request.method == 'POST':
        query = request.form['description']

        if query:
            # Preprocess the query
            preprocessed_query = preprocess_text(query)

            try:
                # Predict using the model
                stock_code_pred, unit_price_pred, country_pred = model.predict(preprocessed_query)
                
                # Decode the predictions
                top_stock_codes = np.argsort(stock_code_pred[0])[::-1][:5]  # Get top 5 stock code predictions
                predicted_unit_price = unit_price_pred[0][0]
                top_countries = np.argsort(country_pred[0])[::-1][:5]  # Get top 5 country predictions

                # Gather product details for top stock codes
                for stock_code_idx in top_stock_codes:
                    stock_code = stock_code_encoder.inverse_transform([stock_code_idx])[0]
                    
                    # Find the product details based on stock code
                    product_rows = product_data[product_data['StockCode'] == stock_code]

                    for _, product_row in product_rows.iterrows():
                        product_details = {
                            "Stock Code": product_row['StockCode'],
                            "Description": product_row['Description'],
                            "Unit Price": product_row['UnitPrice'],
                            "Country": product_row['Country']
                        }
                        products.append(product_details)

                response = "Product details predicted successfully."
            except Exception as e:
                response = f"Error: {str(e)}"
                products = []

    return render_template('text_query.html', products=products, response=response)




@app.route('/image-query', methods=['GET', 'POST'])
def image_query():
    if request.method == 'POST':
        image_file = request.files.get('image_data')
        
        if image_file:
            try:
                # Read the image file and convert it to a numpy array
                image = Image.open(image_file)
                image_np = np.array(image)
                
                # Initialize OCR reader
                reader = easyocr.Reader(['en'])
                
                # Perform OCR on the numpy array image
                result = reader.readtext(image_np)
                
                # Extract text from OCR result
                extracted_text = ' '.join([text for _, text, _ in result])
                
                # Preprocess the extracted text
                preprocessed_query = preprocess_text(extracted_text)
                
                try:
                    # Predict using the model
                    stock_code_pred, unit_price_pred, country_pred = model.predict(preprocessed_query)
                    
                    # Decode the predictions
                    top_stock_codes = np.argsort(stock_code_pred[0])[::-1][:5]  # Get top 5 stock code predictions
                    predicted_unit_price = unit_price_pred[0][0]
                    top_countries = np.argsort(country_pred[0])[::-1][:5]  # Get top 5 country predictions

                    products = []
                    # Gather product details for top stock codes
                    for stock_code_idx in top_stock_codes:
                        stock_code = stock_code_encoder.inverse_transform([stock_code_idx])[0]
                        
                        # Find the product details based on stock code
                        product_rows = product_data[product_data['StockCode'] == stock_code]

                        for _, product_row in product_rows.iterrows():
                            product_details = {
                                "Stock Code": product_row['StockCode'],
                                "Description": product_row['Description'],
                                "Unit Price": product_row['UnitPrice'],
                                "Country": product_row['Country']
                            }
                            products.append(product_details)

                    response = "Product details predicted successfully."
                except Exception as e:
                    response = f"Error: {str(e)}"
                    products = []
            except Exception as e:
                response = f"Error processing image: {str(e)}"
                products = []
        else:
            response = "No image file provided."
            products = []

        return render_template('image_query.html', products=products, response=response, description=extracted_text)

    return render_template('image_query.html')


@app.route('/image-upload', methods=['GET', 'POST'])
def product_description():
    if request.method == 'POST':
        image_file = request.files.get('image_data')
        
        if image_file:
            try:
                # Read the image file and convert it to a numpy array
                image = Image.open(image_file)
                image_np = np.array(image)
                
                # Initialize OCR reader
                reader = easyocr.Reader(['en'])
                
                # Perform OCR on the numpy array image
                result = reader.readtext(image_np)
                
                # Extract text from OCR result
                extracted_text = ' '.join([text for _, text, _ in result])
                
                # Preprocess the extracted text
                preprocessed_query = preprocess_text(extracted_text)
                
                try:
                    # Predict using the model
                    stock_code_pred, unit_price_pred, country_pred = model.predict(preprocessed_query)
                    
                    # Decode the predictions
                    top_stock_codes = np.argsort(stock_code_pred[0])[::-1][:5]  # Get top 5 stock code predictions
                    predicted_unit_price = unit_price_pred[0][0]
                    top_countries = np.argsort(country_pred[0])[::-1][:5]  # Get top 5 country predictions

                    products = []
                    # Gather product details for top stock codes
                    for stock_code_idx in top_stock_codes:
                        stock_code = stock_code_encoder.inverse_transform([stock_code_idx])[0]
                        
                        # Find the product details based on stock code
                        product_rows = product_data[product_data['StockCode'] == stock_code]

                        for _, product_row in product_rows.iterrows():
                            product_details = {
                                "Stock Code": product_row['StockCode'],
                                "Description": product_row['Description'],
                                "Unit Price": product_row['UnitPrice'],
                                "Country": product_row['Country']
                            }
                            products.append(product_details)

                    response = "Product details predicted successfully."
                except Exception as e:
                    response = f"Error: {str(e)}"
                    products = []
            except Exception as e:
                response = f"Error processing image: {str(e)}"
                products = []
        else:
            response = "No image file provided."
            products = []

        return render_template('image_upload.html', products=products, response=response, description=extracted_text)

    return render_template('image_upload.html')


@app.route('/sample_response', methods=['GET'])
def sample_response():
    return render_template('index.html')


#  # Endpoint for product recommendations
# @app.route('/recommend', methods=['POST'])
# def recommend_products():
#     data = request.json
#     query_description = data['query']
#     product_matches = pinecone_service.recommend_products(query_description)
#     return jsonify({
#         'products': product_matches,
#         'response': f"Here are the top products matching your query '{query_description}'"
#     })

# # Endpoint for OCR functionality (for handwritten text)
# @app.route('/ocr-handwritten', methods=['POST'])
# def ocr_handwritten_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400
#     image_file = request.files['image']
#     extracted_text = ocr_service.extract_text(image_file)
#     product_matches = pinecone_service.recommend_products(extracted_text)
#     return jsonify({
#         'extracted_text': extracted_text,
#         'products': product_matches,
#         'response': f"Here are the top products matching your handwritten query"
#     })

# # Endpoint for web scraping product images
# @app.route('/scrape-images', methods=['POST'])
# def scrape_images():
#     data = request.json
#     url = data['url']
#     image_urls = web_scraper_service.scrape_images(url)
#     return jsonify({
#         'image_urls': image_urls,
#         'response': f"Scraped {len(image_urls)} images from the URL '{url}'"
#     })

# # Endpoint for image-based product detection
# @app.route('/detect-product', methods=['POST'])
# def detect_product():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400
#     image_file = request.files['image']
#     class_name = cnn_service.predict_product_class(image_file)
#     product_matches = pinecone_service.recommend_products(class_name)
#     return jsonify({
#         'class_name': class_name,
#         'products': product_matches,
#         'response': f"Detected product class '{class_name}' and here are the top matching products"
#     }) 

if __name__ == '__main__':
    waitress.serve(app, host='0.0.0.0', port=5000)
