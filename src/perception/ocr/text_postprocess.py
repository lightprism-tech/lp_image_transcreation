"""
Text Post-processor
Cleans and normalizes extracted OCR text
"""

import re


class TextPostProcessor:
    """Post-processes OCR output for better quality"""
    
    def __init__(self):
        """Initialize text post-processor"""
        pass
    
    def clean(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters if needed
        # text = re.sub(r'[^\w\s\.,!?-]', '', text)
        
        # Capitalize sentences
        text = text.strip()
        
        return text
    
    def merge_text_blocks(self, text_blocks: list) -> dict:
        """
        Merge and organize multiple text blocks
        
        Args:
            text_blocks: List of text extraction results
            
        Returns:
            Organized text structure:
            {
                'full_text': str,
                'sentences': list,
                'paragraphs': list
            }
        """
        # TODO: Implement smart text merging
        # Consider spatial relationships to group text
        
        all_text = ' '.join([block.get('text', '') for block in text_blocks])
        cleaned = self.clean(all_text)
        
        return {
            'full_text': cleaned,
            'sentences': [s.strip() for s in re.split(r'[.!?]+', cleaned) if s.strip()],
            'paragraphs': [cleaned]
        }
