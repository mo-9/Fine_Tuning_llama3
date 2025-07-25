import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict
import time
import random

class WebScraper:
    def __init__(self, delay_range=(1, 3)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def scrape_url(self, url: str) -> Dict:
        """Scrape content from a single URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            text = soup.get_text(separator=' ', strip=True)
            
            title = soup.find('title')
            title = title.text.strip() if title else ""
            
            meta_description = soup.find('meta', attrs={'name': 'description'})
            description = meta_description.get('content', '') if meta_description else ""
            
            return {
                'url': url,
                'title': title,
                'description': description,
                'content': text,
                'source': 'web_scraping',
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def scrape_urls(self, urls: List[str]) -> List[Dict]:
        results = []
        
        for url in urls:
            result = self.scrape_url(url)
            if result:
                results.append(result)
            
            delay = random.uniform(*self.delay_range)
            time.sleep(delay)
            
        return results
    
    def search_and_scrape(self, query: str, num_results: int = 10) -> List[Dict]:

        search_urls = [
            f"https://example.com/search?q={query}",
            # Add more search sources
        ]
        
        return self.scrape_urls(search_urls[:num_results])

