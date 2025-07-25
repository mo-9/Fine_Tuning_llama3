import re
import hashlib
import logging
from typing import List, Dict, Set
import pandas as pd

class DataCleaner:
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
            
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[^\w\s.,!?;:\-\(\)]', '', text)
        
        text = text.strip()
        
        return text
    
    def is_duplicate(self, text: str) -> bool:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def quality_filter(self, text: str, min_length: int = 50, max_length: int = 10000) -> bool:
        """Filter text based on quality criteria."""
        if not text:
            return False
            
        # Length filter
        if len(text) < min_length or len(text) > max_length:
            return False
            
        # Language detection (simplified - check for common English words)
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        word_count = sum(1 for word in english_words if word in text.lower())
        
        if word_count < 2:  # Must contain at least 2 common English words
            return False
            
        return True
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        cleaned_documents = []
        
        for doc in documents:
            if not doc or 'content' not in doc:
                continue
                
            # Clean text
            cleaned_text = self.clean_text(doc['content'])
            
            if self.is_duplicate(cleaned_text):
                logging.info(f"Skipping duplicate document: {doc.get('url', 'unknown')}")
                continue
                
            if not self.quality_filter(cleaned_text):
                logging.info(f"Skipping low-quality document: {doc.get('url', 'unknown')}")
                continue
                
            doc['content'] = cleaned_text
            doc['processed'] = True
            cleaned_documents.append(doc)
            
        logging.info(f"Processed {len(documents)} documents, kept {len(cleaned_documents)}")
        return cleaned_documents

