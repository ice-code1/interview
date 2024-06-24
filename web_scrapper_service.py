import requests
from bs4 import BeautifulSoup

class WebScraperService:
    def __init__(self):
        pass

    def scrape_images(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all('img')

        image_urls = []
        for img in images:
            src = img.get('src')
            if src and (src.endswith('.jpg') or src.endswith('.png')):
                image_urls.append(src)
        
        return image_urls
