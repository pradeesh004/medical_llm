import easyocr
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, languages=['en']):
        """Initialize OCR processor with specified languages."""
        self.reader = easyocr.Reader(languages)
        logger.info(f"Initialized OCR processor with languages: {languages}")

    def extract_text_from_input(self, input_path):
        """
        Extract text from an image or PDF document using EasyOCR.
        
        Args:
            input_path (str or Path): Path to the input image or PDF file
            
        Returns:
            dict: Dictionary containing extracted text and metadata
        """
        input_path = Path(input_path)
        try:
            total_text = []
            metadata = {
                'page_count': 0,
                'processing_time': datetime.now().isoformat(),
                'file_type': input_path.suffix.lower(),
                'filename': input_path.name
            }

            # Process image files
            if input_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif'}:
                logger.info(f"Processing image file: {input_path.name}")
                results = self.reader.readtext(str(input_path))
                text = '\n'.join(result[1] for result in results)
                total_text.append(text)
                metadata['page_count'] = 1

            # Process PDF files
            elif input_path.suffix.lower() == '.pdf':
                logger.info(f"Processing PDF file: {input_path.name}")
                pages = convert_from_path(str(input_path), 500)
                metadata['page_count'] = len(pages)
                
                for page_num, page in enumerate(pages, 1):
                    logger.info(f"Processing page {page_num} of {len(pages)}")
                    page_np = np.array(page)
                    results = self.reader.readtext(page_np)
                    page_text = '\n'.join(result[1] for result in results)
                    total_text.append(f"--- Page {page_num} ---\n{page_text}")

            else:
                raise ValueError(f"Unsupported file type: {input_path.suffix}")

            # Combine results
            combined_text = '\n\n'.join(total_text)
            metadata['char_count'] = len(combined_text)
            metadata['word_count'] = len(combined_text.split())

            logger.info(f"Successfully processed {input_path.name}")
            return {
                'text': combined_text,
                'metadata': metadata,
                'success': True
            }

        except Exception as e:
            logger.error(f"Error processing file {input_path}: {str(e)}")
            return {
                'text': '',
                'metadata': {},
                'success': False,
                'error': str(e)
            }

    def extract_text_with_positions(self, input_path):
        """
        Extract text with position information from an image file.
        
        Args:
            input_path (str or Path): Path to the input image file
            
        Returns:
            list: List of dictionaries containing text and position information
        """
        try:
            input_path = Path(input_path)
            if input_path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.gif'}:
                raise ValueError("This method only supports image files")

            results = self.reader.readtext(str(input_path))
            structured_results = []

            for bbox, text, conf in results:
                # Convert bbox points to a more useful format
                top_left, top_right, bottom_right, bottom_left = bbox
                structured_results.append({
                    'text': text,
                    'confidence': conf,
                    'position': {
                        'top_left': {'x': top_left[0], 'y': top_left[1]},
                        'top_right': {'x': top_right[0], 'y': top_right[1]},
                        'bottom_right': {'x': bottom_right[0], 'y': bottom_right[1]},
                        'bottom_left': {'x': bottom_left[0], 'y': bottom_left[1]}
                    }
                })

            return structured_results

        except Exception as e:
            logger.error(f"Error processing file {input_path} with positions: {str(e)}")
            return []

    def is_medical_document(self, text):
        """
        Basic check if the extracted text appears to be from a medical document.
        
        Args:
            text (str): Extracted text to analyze
            
        Returns:
            bool: True if the text appears to be medical in nature
        """
        medical_keywords = {
            'patient', 'diagnosis', 'treatment', 'hospital', 'doctor',
            'medical', 'clinical', 'symptoms', 'examination', 'results',
            'lab', 'test', 'medication', 'prescription', 'radiography',
            'mri', 'ct scan', 'x-ray', 'ultrasound'
        }
        
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in medical_keywords if keyword in text_lower)
        
        # Consider it medical if it contains at least 3 medical keywords
        return found_keywords >= 3