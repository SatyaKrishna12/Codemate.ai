"""
Sophisticated text processing utilities for document analysis and preprocessing.
Includes normalization, encoding handling, language detection, and quality assessment.
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple, List
import chardet
from enum import Enum

from app.core.logging import get_logger
from app.core.exceptions import ProcessingError

logger = get_logger(__name__)


class TextQuality(Enum):
    """Text quality levels."""
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    POOR = "poor"


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    UNKNOWN = "unknown"


class TextPreprocessor:
    """Advanced text preprocessing utilities."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.min_text_length = 10
        self.max_whitespace_ratio = 0.5
        self.min_alpha_ratio = 0.3
        
        # Language detection patterns (simple regex-based for now)
        self.language_patterns = {
            Language.ENGLISH: re.compile(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', re.IGNORECASE),
            Language.SPANISH: re.compile(r'\b(el|la|los|las|y|o|pero|en|de|con|por|para)\b', re.IGNORECASE),
            Language.FRENCH: re.compile(r'\b(le|la|les|et|ou|mais|dans|de|avec|par|pour)\b', re.IGNORECASE),
            Language.GERMAN: re.compile(r'\b(der|die|das|und|oder|aber|in|von|mit|für|zu)\b', re.IGNORECASE),
            Language.CHINESE: re.compile(r'[\u4e00-\u9fff]+'),
            Language.ARABIC: re.compile(r'[\u0600-\u06ff]+'),
        }
    
    def detect_encoding(self, raw_data: bytes) -> str:
        """
        Detect text encoding from raw bytes.
        
        Args:
            raw_data: Raw bytes from file
            
        Returns:
            Detected encoding
        """
        try:
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0.0)
            
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            
            # Fallback to utf-8 if confidence is too low
            if confidence < 0.7:
                logger.warning(f"Low confidence encoding detection, falling back to utf-8")
                return 'utf-8'
                
            return encoding
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {str(e)}, using utf-8")
            return 'utf-8'
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning whitespace and special characters.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        try:
            # Unicode normalization
            text = unicodedata.normalize('NFKC', text)
            
            # Remove control characters except tabs and newlines
            text = ''.join(char for char in text if not unicodedata.category(char).startswith('C') or char in '\t\n')
            
            # Normalize whitespace
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
            text = re.sub(r'\n{3,}', '\n\n', text)  # More than 2 newlines to 2
            
            # Remove leading/trailing whitespace from lines
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)
            
            # Final cleanup
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Text normalization failed: {str(e)}")
            raise ProcessingError(f"Failed to normalize text: {str(e)}")
    
    def detect_language(self, text: str) -> Language:
        """
        Detect the primary language of the text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language
        """
        if not text or len(text.strip()) < 10:
            return Language.UNKNOWN
        
        # Sample text for analysis (first 1000 chars for performance)
        sample = text[:1000].lower()
        
        language_scores = {}
        
        for language, pattern in self.language_patterns.items():
            matches = pattern.findall(sample)
            score = len(matches)
            if score > 0:
                language_scores[language] = score
        
        if not language_scores:
            return Language.UNKNOWN
        
        # Return language with highest score
        detected_language = max(language_scores, key=language_scores.get)
        
        logger.debug(f"Detected language: {detected_language.value} (scores: {language_scores})")
        
        return detected_language
    
    def assess_text_quality(self, text: str) -> Tuple[TextQuality, Dict[str, float]]:
        """
        Assess the quality of text content.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (quality_level, metrics_dict)
        """
        if not text:
            return TextQuality.POOR, {"length": 0, "alpha_ratio": 0, "whitespace_ratio": 1}
        
        metrics = {}
        
        # Basic metrics
        total_length = len(text)
        alpha_chars = sum(1 for char in text if char.isalpha())
        whitespace_chars = sum(1 for char in text if char.isspace())
        
        metrics["length"] = total_length
        metrics["alpha_ratio"] = alpha_chars / total_length if total_length > 0 else 0
        metrics["whitespace_ratio"] = whitespace_chars / total_length if total_length > 0 else 0
        
        # Word-based metrics
        words = text.split()
        metrics["word_count"] = len(words)
        metrics["avg_word_length"] = sum(len(word) for word in words) / len(words) if words else 0
        
        # Sentence-based metrics
        sentences = re.split(r'[.!?]+', text)
        metrics["sentence_count"] = len([s for s in sentences if s.strip()])
        metrics["avg_sentence_length"] = metrics["word_count"] / metrics["sentence_count"] if metrics["sentence_count"] > 0 else 0
        
        # Special character analysis
        special_chars = sum(1 for char in text if not (char.isalnum() or char.isspace()))
        metrics["special_char_ratio"] = special_chars / total_length if total_length > 0 else 0
        
        # Determine quality level
        quality = self._calculate_quality_score(metrics)
        
        logger.debug(f"Text quality assessment: {quality.value} (metrics: {metrics})")
        
        return quality, metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> TextQuality:
        """Calculate overall quality score from metrics."""
        
        # Poor quality conditions
        if (metrics["length"] < self.min_text_length or
            metrics["alpha_ratio"] < self.min_alpha_ratio or
            metrics["whitespace_ratio"] > self.max_whitespace_ratio):
            return TextQuality.POOR
        
        # High quality conditions
        if (metrics["length"] > 100 and
            metrics["alpha_ratio"] > 0.7 and
            metrics["avg_word_length"] > 3 and
            metrics["avg_sentence_length"] > 5 and
            metrics["special_char_ratio"] < 0.1):
            return TextQuality.HIGH
        
        # Medium quality conditions
        if (metrics["length"] > 50 and
            metrics["alpha_ratio"] > 0.5 and
            metrics["avg_word_length"] > 2):
            return TextQuality.MEDIUM
        
        # Default to low quality
        return TextQuality.LOW
    
    def clean_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        Clean text while optionally preserving document structure.
        
        Args:
            text: Input text
            preserve_structure: Whether to preserve headers, lists, etc.
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        try:
            # Basic normalization
            cleaned_text = self.normalize_text(text)
            
            if preserve_structure:
                # Preserve markdown-style headers
                cleaned_text = re.sub(r'^(#{1,6})\s+(.+)$', r'\1 \2', cleaned_text, flags=re.MULTILINE)
                
                # Preserve bullet points
                cleaned_text = re.sub(r'^(\s*[-*+•])\s+(.+)$', r'\1 \2', cleaned_text, flags=re.MULTILINE)
                
                # Preserve numbered lists
                cleaned_text = re.sub(r'^(\s*\d+\.)\s+(.+)$', r'\1 \2', cleaned_text, flags=re.MULTILINE)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            raise ProcessingError(f"Failed to clean text: {str(e)}")
    
    def preprocess_document(self, raw_content: str, file_type: str = "txt") -> Dict[str, any]:
        """
        Complete preprocessing pipeline for document content.
        
        Args:
            raw_content: Raw document content
            file_type: Type of document for context
            
        Returns:
            Dictionary with processed content and metadata
        """
        try:
            # Clean and normalize text
            cleaned_text = self.clean_text(raw_content, preserve_structure=True)
            
            # Assess quality
            quality, quality_metrics = self.assess_text_quality(cleaned_text)
            
            # Detect language
            language = self.detect_language(cleaned_text)
            
            # Skip poor quality text
            if quality == TextQuality.POOR:
                logger.warning(f"Poor quality text detected, metrics: {quality_metrics}")
                return {
                    "processed_text": "",
                    "skipped": True,
                    "skip_reason": "poor_quality",
                    "quality": quality.value,
                    "quality_metrics": quality_metrics,
                    "language": language.value
                }
            
            logger.info(f"Text preprocessing completed: quality={quality.value}, language={language.value}")
            
            return {
                "processed_text": cleaned_text,
                "skipped": False,
                "quality": quality.value,
                "quality_metrics": quality_metrics,
                "language": language.value,
                "original_length": len(raw_content),
                "processed_length": len(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Document preprocessing failed: {str(e)}")
            raise ProcessingError(f"Failed to preprocess document: {str(e)}")


# Global instance for easy access
text_preprocessor = TextPreprocessor()
