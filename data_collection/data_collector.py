import logging
from typing import List, Dict
from .web_scraper import WebScraper
from .pdf_extractor import PDFExtractor
from ..data_processing.data_cleaner import DataCleaner
from ..data_processing.data_storage import DataStorage

class DataCollector:
    def __init__(self, storage_path: str = "data/processed_data.db"):
        self.web_scraper = WebScraper()
        self.pdf_extractor = PDFExtractor()
        self.data_cleaner = DataCleaner()
        self.data_storage = DataStorage(storage_path)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def collect_web_data(self, urls: List[str]) -> List[Dict]:
        """Collect data from web URLs."""
        self.logger.info(f"Starting web data collection for {len(urls)} URLs")
        
        raw_documents = self.web_scraper.scrape_urls(urls)
        cleaned_documents = self.data_cleaner.process_documents(raw_documents)
        
        if cleaned_documents:
            document_ids = self.data_storage.store_documents(cleaned_documents)
            self.logger.info(f"Stored {len(document_ids)} web documents")
        
        return cleaned_documents
    
    def collect_pdf_data(self, pdf_paths: List[str]) -> List[Dict]:
        """Collect data from PDF files."""
        self.logger.info(f"Starting PDF data collection for {len(pdf_paths)} files")
        
        raw_documents = []
        for pdf_path in pdf_paths:
            doc_data = self.pdf_extractor.extract_text_and_layout(pdf_path)
            if doc_data:
                # Add metadata
                metadata = self.pdf_extractor.extract_metadata(pdf_path)
                doc_data['metadata'] = metadata
                raw_documents.append(doc_data)
        
        cleaned_documents = self.data_cleaner.process_documents(raw_documents)
        
        if cleaned_documents:
            document_ids = self.data_storage.store_documents(cleaned_documents)
            self.logger.info(f"Stored {len(document_ids)} PDF documents")
        
        return cleaned_documents
    
    def collect_domain_data(self, domain: str, num_web_results: int = 20) -> List[Dict]:
        """Collect domain-specific data using search and scraping."""
        self.logger.info(f"Collecting domain-specific data for: {domain}")
        
        # This is a simplified implementation
        # In practice, you'd use more sophisticated search methods
        search_results = self.web_scraper.search_and_scrape(domain, num_web_results)
        cleaned_documents = self.data_cleaner.process_documents(search_results)
        
        if cleaned_documents:
            document_ids = self.data_storage.store_documents(cleaned_documents)
            self.logger.info(f"Stored {len(document_ids)} domain-specific documents")
        
        return cleaned_documents
    
    def get_stored_documents(self, limit: int = None) -> List[Dict]:
        """Retrieve stored documents from the database."""
        return self.data_storage.get_documents(limit)

