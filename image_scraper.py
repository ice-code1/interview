import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

class WebScraperService:
    def __init__(self):
        pass

    def scrape_product_data(self, url, download_folder='images'):
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Adjust these based on the actual HTML structure of Jumia's product listings
        products = soup.find_all('article', class_='prd _fb col c-prd')

        data = []
        for i, product in enumerate(products):
            img_tag = product.find('img')
            if img_tag:
                img_url = img_tag.get('data-src') or img_tag.get('src')
                class_name = product.get('data-category', 'unknown')  # Adjust this if necessary
                
                # Download the image
                img_data = requests.get(img_url).content
                img_path = os.path.join(download_folder, f'image_{i}.jpg')
                with open(img_path, 'wb') as handler:
                    handler.write(img_data)
                
                data.append({'image_path': img_path, 'class': class_name})

        return data

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

# Example usage
if __name__ == '__main__':
    scraper = WebScraperService()
    url = 'https://www.jumia.com.ng/home-office/'  # Example URL
    scraped_data = scraper.scrape_product_data(url)
    scraper.save_to_csv(scraped_data, 'CNN_Model_Train_Data.csv')
