import pytesseract
from PIL import Image

class OCRService:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your Tesseract executable path
        

    def extract_text(self, image_file):
        image = Image.open(image_file)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text
