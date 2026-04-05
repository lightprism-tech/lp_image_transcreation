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

    def summarize_styles(self, text_blocks: list) -> dict:
        """
        Summarize typography information from OCR style metadata.

        Returns:
            {
                "font_families": [{"name": str, "count": int}],
                "font_weights": {"normal": int, "bold": int},
                "avg_font_size": float,
                "styled_regions": int
            }
        """
        family_counts = {}
        weight_counts = {"normal": 0, "bold": 0}
        font_sizes = []

        for block in text_blocks or []:
            style = block.get("style") or {}
            if not isinstance(style, dict) or not style:
                continue

            family = str(style.get("font_family", "")).strip()
            if family:
                family_counts[family] = family_counts.get(family, 0) + 1

            weight = str(style.get("font_weight", "normal")).strip().lower()
            if weight not in weight_counts:
                weight_counts[weight] = 0
            weight_counts[weight] += 1

            size = style.get("font_size")
            if isinstance(size, (int, float)):
                font_sizes.append(float(size))

        ordered_families = sorted(
            [{"name": k, "count": v} for k, v in family_counts.items()],
            key=lambda item: item["count"],
            reverse=True,
        )
        avg_size = round(sum(font_sizes) / len(font_sizes), 2) if font_sizes else 0.0

        return {
            "font_families": ordered_families,
            "font_weights": weight_counts,
            "avg_font_size": avg_size,
            "styled_regions": sum(weight_counts.values()),
        }
