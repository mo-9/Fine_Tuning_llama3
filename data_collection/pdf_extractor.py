import pdfplumber
import logging
from typing import List, Dict

class PDFExtractor:
    def __init__(self):
        pass

    def extract_text_and_layout(self, pdf_path: str) -> Dict:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_text = []
                pages_data = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    full_text.append(text if text else "")
                    
                    # Extract layout information (simplified for now)
                    # This can be expanded to include more detailed layout data
                    words = page.extract_words()
                    
                    pages_data.append({
                        "page_number": i + 1,
                        "text": text,
                        "words": words, # Example of layout data
                        "width": page.width,
                        "height": page.height,
                    })
            
            return {
                "pdf_path": pdf_path,
                "full_text": "\n".join(full_text),
                "pages_data": pages_data,
                "source": "pdf_extraction",
                "timestamp": ""
            }
        except Exception as e:
            logging.error(f"Error extracting from {pdf_path}: {str(e)}")
            return None

    def extract_metadata(self, pdf_path: str) -> Dict:
        """Extracts metadata from a PDF."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = pdf.metadata
                return metadata
        except Exception as e:
            logging.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
            return None


