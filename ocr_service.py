import pytesseract
from PIL import Image

class OCRService:
    def __init__(self):
        pass

    def extract_text(self, image_file):
        image = Image.open(image_file)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text
