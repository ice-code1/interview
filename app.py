from flask import Flask, jsonify, request
from pinecone_service import PineconeService
from ocr_service import OCRService
from web_scraper_service import WebScraperService
from cnn_service import CNNService

app = Flask(__name__)

# Initialize services with your API key and other necessary parameters
pinecone_service = PineconeService(
    api_key='21f2127c-159b-4a3b-9fbc-a9b3a1ec422c',
    index_name='product-vectors',
    dataset_filename='cleaned_dataset.csv'
)
ocr_service = OCRService()
web_scraper_service = WebScraperService()
cnn_service = CNNService('cnn_model.h5')

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
    app.run(debug=True)
