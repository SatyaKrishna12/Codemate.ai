"""
Post-Processing Service for snippet generation, passage highlighting, and key-quote extraction.

This service enhances search results with formatted snippets, highlighted passages,
extracted key quotes with proper attribution, and creates user-friendly presentation
of the retrieved information.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import RankedResult, Snippet, Quote

logger = get_logger(__name__)


@dataclass
class HighlightMatch:
    """Represents a highlighted text match."""
    start: int
    end: int
    text: str
    match_type: str  # "exact", "partial", "semantic"
    relevance_score: float


class TextProcessor:
    """Handles text processing operations for snippets and highlighting."""
    
    def __init__(self):
        """Initialize text processor."""
        self.sentence_patterns = [
            r'(?<=[.!?])\s+',  # Split on sentence endings
            r'(?<=\.)\s+(?=[A-Z])',  # Split on periods followed by capital letters
        ]
        
        # Common stop words to exclude from key phrase extraction
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'will', 'about', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
            'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'has',
            'two', 'more', 'very', 'after', 'words', 'long', 'than', 'first',
            'been', 'call', 'who', 'its', 'now', 'find', 'could', 'made', 'may',
            'part', 'over', 'new', 'sound', 'take', 'only', 'little', 'work',
            'know', 'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very'
        }
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Clean text first
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text."""
        # Convert to lowercase and remove punctuation
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Remove stop words
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Extract n-grams (1-3 words)
        phrases = []
        
        # 1-grams (individual words)
        phrases.extend(filtered_words)
        
        # 2-grams
        for i in range(len(filtered_words) - 1):
            phrase = f"{filtered_words[i]} {filtered_words[i+1]}"
            phrases.append(phrase)
        
        # 3-grams
        for i in range(len(filtered_words) - 2):
            phrase = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
            phrases.append(phrase)
        
        # Count frequency and return most common
        from collections import Counter
        phrase_counts = Counter(phrases)
        
        # Return top phrases (prefer longer phrases)
        top_phrases = []
        for phrase, count in phrase_counts.most_common():
            if len(phrase.split()) > 1 and count > 1:  # Multi-word phrases with frequency > 1
                top_phrases.append(phrase)
            elif len(phrase.split()) == 1 and count > 2:  # Single words with frequency > 2
                top_phrases.append(phrase)
            
            if len(top_phrases) >= max_phrases:
                break
        
        return top_phrases
    
    def find_best_sentences(
        self, 
        text: str, 
        query_terms: List[str], 
        max_sentences: int = 3
    ) -> List[str]:
        """Find the most relevant sentences based on query terms."""
        sentences = self.split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Score sentences based on query term coverage
        sentence_scores = []
        for sentence in sentences:
            score = self._score_sentence_relevance(sentence, query_terms)
            sentence_scores.append((sentence, score))
        
        # Sort by score and return top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in sentence_scores[:max_sentences]]
    
    def _score_sentence_relevance(self, sentence: str, query_terms: List[str]) -> float:
        """Score a sentence's relevance to query terms."""
        sentence_lower = sentence.lower()
        score = 0.0
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Exact match bonus
            if term_lower in sentence_lower:
                score += 2.0
            
            # Partial match bonus
            for word in term_lower.split():
                if word in sentence_lower:
                    score += 0.5
        
        # Length normalization (prefer moderate length sentences)
        length_score = 1.0
        if 50 <= len(sentence) <= 200:
            length_score = 1.2
        elif len(sentence) < 20:
            length_score = 0.5
        
        return score * length_score


class SnippetGenerator:
    """Generates formatted snippets from search results."""
    
    def __init__(self):
        """Initialize snippet generator."""
        self.text_processor = TextProcessor()
        self.max_snippet_length = getattr(settings, 'snippet_max_chars', 300)
    
    def generate_snippet(
        self, 
        content: str, 
        query: str,
        context_chars: int = 50
    ) -> Snippet:
        """
        Generate a formatted snippet from content.
        
        Args:
            content: Full content text
            query: Search query for relevance
            context_chars: Characters of context around matches
            
        Returns:
            Formatted snippet with highlighting
        """
        try:
            query_terms = self._extract_query_terms(query)
            
            # Find best sentences
            best_sentences = self.text_processor.find_best_sentences(
                content, query_terms, max_sentences=2
            )
            
            if not best_sentences:
                # Fallback: use first part of content
                snippet_text = content[:self.max_snippet_length]
                if len(content) > self.max_snippet_length:
                    snippet_text += "..."
            else:
                # Join best sentences
                snippet_text = " ".join(best_sentences)
                if len(snippet_text) > self.max_snippet_length:
                    snippet_text = snippet_text[:self.max_snippet_length] + "..."
            
            # Find highlight positions
            highlights = self._find_highlight_positions(snippet_text, query_terms)
            
            return Snippet(
                text=snippet_text,
                highlighted_terms=query_terms,
                character_count=len(snippet_text),
                highlight_positions=highlights,
                relevance_score=self._calculate_snippet_relevance(snippet_text, query_terms)
            )
            
        except Exception as e:
            logger.warning(f"Error generating snippet: {e}")
            # Return basic snippet
            return Snippet(
                text=content[:self.max_snippet_length] + ("..." if len(content) > self.max_snippet_length else ""),
                highlighted_terms=[],
                character_count=min(len(content), self.max_snippet_length),
                highlight_positions=[],
                relevance_score=0.5
            )
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract search terms from query."""
        # Remove common operators and split
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        terms = [term.strip() for term in clean_query.split() if len(term.strip()) > 2]
        return terms
    
    def _find_highlight_positions(self, text: str, terms: List[str]) -> List[Dict[str, Any]]:
        """Find positions of terms to highlight in text."""
        highlight_positions = []
        text_lower = text.lower()
        
        for term in terms:
            term_lower = term.lower()
            start = 0
            
            while True:
                pos = text_lower.find(term_lower, start)
                if pos == -1:
                    break
                
                highlight_positions.append({
                    'start': pos,
                    'end': pos + len(term),
                    'term': term,
                    'match_type': 'exact'
                })
                
                start = pos + 1
        
        # Sort by position
        highlight_positions.sort(key=lambda x: x['start'])
        return highlight_positions
    
    def _calculate_snippet_relevance(self, snippet: str, query_terms: List[str]) -> float:
        """Calculate relevance score for a snippet."""
        snippet_lower = snippet.lower()
        score = 0.0
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Count exact matches
            exact_matches = snippet_lower.count(term_lower)
            score += exact_matches * 0.5
            
            # Count word matches
            for word in term_lower.split():
                word_matches = snippet_lower.count(word)
                score += word_matches * 0.2
        
        # Normalize by snippet length
        normalized_score = score / (len(snippet) / 100)  # Per 100 characters
        return min(normalized_score, 1.0)


class QuoteExtractor:
    """Extracts key quotes from search results."""
    
    def __init__(self):
        """Initialize quote extractor."""
        self.text_processor = TextProcessor()
        
        # Patterns that indicate quotable content
        self.quote_patterns = [
            r'"([^"]+)"',  # Text in quotes
            r'according to [^,]+,?\s*"?([^"\.]+)"?',  # According to X
            r'research shows that ([^\.]+)',  # Research shows that
            r'studies indicate that ([^\.]+)',  # Studies indicate that
            r'expert[s]? say[s]? that ([^\.]+)',  # Experts say that
            r'data shows that ([^\.]+)',  # Data shows that
        ]
    
    def extract_quotes(
        self, 
        content: str, 
        query: str,
        max_quotes: int = 3
    ) -> List[Quote]:
        """
        Extract key quotes from content.
        
        Args:
            content: Content to extract quotes from
            query: Search query for relevance
            max_quotes: Maximum number of quotes to extract
            
        Returns:
            List of extracted quotes with attribution
        """
        quotes = []
        query_terms = self._extract_query_terms(query)
        
        try:
            # Method 1: Extract explicit quotes (in quotation marks)
            explicit_quotes = self._extract_explicit_quotes(content, query_terms)
            quotes.extend(explicit_quotes)
            
            # Method 2: Extract statement-based quotes
            statement_quotes = self._extract_statement_quotes(content, query_terms)
            quotes.extend(statement_quotes)
            
            # Method 3: Extract key sentences as quotes
            if len(quotes) < max_quotes:
                sentence_quotes = self._extract_sentence_quotes(content, query_terms)
                quotes.extend(sentence_quotes)
            
            # Remove duplicates and score quotes
            unique_quotes = self._deduplicate_quotes(quotes)
            scored_quotes = self._score_quotes(unique_quotes, query_terms)
            
            # Return top quotes
            return scored_quotes[:max_quotes]
            
        except Exception as e:
            logger.warning(f"Error extracting quotes: {e}")
            return []
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract search terms from query."""
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        terms = [term.strip() for term in clean_query.split() if len(term.strip()) > 2]
        return terms
    
    def _extract_explicit_quotes(self, content: str, query_terms: List[str]) -> List[Quote]:
        """Extract text that appears in quotation marks."""
        quotes = []
        
        # Find text in quotation marks
        quote_matches = re.finditer(r'"([^"]+)"', content)
        
        for match in quote_matches:
            quote_text = match.group(1).strip()
            if len(quote_text) > 20:  # Filter very short quotes
                # Try to find attribution (author/source before or after)
                attribution = self._find_attribution(content, match.start(), match.end())
                
                quote = Quote(
                    text=quote_text,
                    source_reference="Direct quote",
                    attribution=attribution,
                    confidence_score=0.8,  # High confidence for explicit quotes
                    context_before=content[max(0, match.start()-50):match.start()],
                    context_after=content[match.end():match.end()+50],
                    relevance_to_query=self._calculate_quote_relevance(quote_text, query_terms)
                )
                quotes.append(quote)
        
        return quotes
    
    def _extract_statement_quotes(self, content: str, query_terms: List[str]) -> List[Quote]:
        """Extract authoritative statements as quotes."""
        quotes = []
        
        for pattern in self.quote_patterns[1:]:  # Skip the first pattern (explicit quotes)
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                if len(match.groups()) > 0:
                    quote_text = match.group(1).strip()
                    if len(quote_text) > 15:
                        # Extract the full match as attribution context
                        full_match = match.group(0)
                        attribution = full_match.replace(quote_text, "").strip()
                        
                        quote = Quote(
                            text=quote_text,
                            source_reference="Extracted statement",
                            attribution=attribution,
                            confidence_score=0.6,
                            context_before=content[max(0, match.start()-30):match.start()],
                            context_after=content[match.end():match.end()+30],
                            relevance_to_query=self._calculate_quote_relevance(quote_text, query_terms)
                        )
                        quotes.append(quote)
        
        return quotes
    
    def _extract_sentence_quotes(self, content: str, query_terms: List[str]) -> List[Quote]:
        """Extract key sentences as quotes."""
        quotes = []
        sentences = self.text_processor.split_into_sentences(content)
        
        # Score sentences and take the best ones
        sentence_scores = []
        for sentence in sentences:
            relevance = self._calculate_quote_relevance(sentence, query_terms)
            if relevance > 0.3 and len(sentence) > 30:  # Filter low relevance and short sentences
                sentence_scores.append((sentence, relevance))
        
        # Sort by relevance and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        for sentence, relevance in sentence_scores[:2]:  # Top 2 sentences
            quote = Quote(
                text=sentence,
                source_reference="Key sentence",
                attribution="",
                confidence_score=0.4,  # Lower confidence for sentence extraction
                context_before="",
                context_after="",
                relevance_to_query=relevance
            )
            quotes.append(quote)
        
        return quotes
    
    def _find_attribution(self, content: str, quote_start: int, quote_end: int) -> str:
        """Find attribution for a quote by looking at surrounding context."""
        # Look before the quote
        before_context = content[max(0, quote_start-100):quote_start]
        after_context = content[quote_end:quote_end+100]
        
        # Patterns for attribution
        attribution_patterns = [
            r'(Dr\.?\s+[A-Z][a-z]+)',  # Dr. Name
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:said|stated|explained|noted)',  # Name said
            r'according to\s+([^,\.]+)',  # according to X
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),?\s+(?:a|an)\s+([^,\.]+)',  # Name, a title
        ]
        
        # Check before context first, then after
        for context in [before_context, after_context]:
            for pattern in attribution_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return ""
    
    def _deduplicate_quotes(self, quotes: List[Quote]) -> List[Quote]:
        """Remove duplicate quotes."""
        seen_texts = set()
        unique_quotes = []
        
        for quote in quotes:
            # Normalize text for comparison
            normalized_text = re.sub(r'\s+', ' ', quote.text.lower().strip())
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_quotes.append(quote)
        
        return unique_quotes
    
    def _score_quotes(self, quotes: List[Quote], query_terms: List[str]) -> List[Quote]:
        """Score and sort quotes by relevance and quality."""
        for quote in quotes:
            # Combine relevance and confidence
            combined_score = (quote.relevance_to_query * 0.7) + (quote.confidence_score * 0.3)
            
            # Bonus for having attribution
            if quote.attribution:
                combined_score += 0.1
            
            # Update the quote with combined score
            quote.confidence_score = combined_score
        
        # Sort by combined score
        quotes.sort(key=lambda x: x.confidence_score, reverse=True)
        return quotes
    
    def _calculate_quote_relevance(self, quote_text: str, query_terms: List[str]) -> float:
        """Calculate relevance of quote to query terms."""
        quote_lower = quote_text.lower()
        relevance = 0.0
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Exact term match
            if term_lower in quote_lower:
                relevance += 0.3
            
            # Word matches
            for word in term_lower.split():
                if word in quote_lower:
                    relevance += 0.1
        
        return min(relevance, 1.0)


class PostProcessingService:
    """
    Service for post-processing search results with snippet generation,
    passage highlighting, and key quote extraction.
    """
    
    def __init__(self):
        """Initialize post-processing service."""
        self.snippet_generator = SnippetGenerator()
        self.quote_extractor = QuoteExtractor()
        self.text_processor = TextProcessor()
    
    def process_results(
        self, 
        results: List[RankedResult],
        query: str,
        generate_snippets: bool = True,
        extract_quotes: bool = True,
        highlight_terms: bool = True
    ) -> List[RankedResult]:
        """
        Process search results with enhanced formatting and extraction.
        
        Args:
            results: List of search results to process
            query: Original search query
            generate_snippets: Whether to generate formatted snippets
            extract_quotes: Whether to extract key quotes
            highlight_terms: Whether to highlight search terms
            
        Returns:
            Enhanced results with snippets, quotes, and highlighting
        """
        if not results:
            return results
        
        try:
            logger.info(f"Post-processing {len(results)} search results")
            
            processed_results = []
            query_terms = self._extract_query_terms(query)
            
            for result in results:
                # Create a copy to avoid modifying original
                processed_result = result.copy() if hasattr(result, 'copy') else result
                
                # Generate snippet if requested
                if generate_snippets:
                    snippet = self.snippet_generator.generate_snippet(
                        result.content_snippet, query
                    )
                    processed_result.relevance_explanation['snippet'] = snippet.__dict__
                
                # Extract quotes if requested
                if extract_quotes:
                    quotes = self.quote_extractor.extract_quotes(
                        result.content_snippet, query, max_quotes=2
                    )
                    processed_result.relevance_explanation['key_quotes'] = [
                        quote.__dict__ for quote in quotes
                    ]
                
                # Highlight terms if requested
                if highlight_terms:
                    highlighted_content = self._highlight_terms(
                        result.content_snippet, query_terms
                    )
                    processed_result.relevance_explanation['highlighted_content'] = highlighted_content
                
                # Extract and add matched terms
                matched_terms = self._find_matched_terms(result.content_snippet, query_terms)
                processed_result.matched_terms = matched_terms
                
                # Add processing metadata
                processed_result.relevance_explanation['post_processing'] = {
                    'has_snippet': generate_snippets,
                    'has_quotes': extract_quotes and len(quotes) > 0 if extract_quotes else False,
                    'has_highlighting': highlight_terms,
                    'matched_terms_count': len(matched_terms),
                    'processing_timestamp': str(datetime.now())
                }
                
                processed_results.append(processed_result)
            
            logger.info(f"Post-processing completed for {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return results  # Return original results if processing fails
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract search terms from query."""
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        terms = [term.strip() for term in clean_query.split() if len(term.strip()) > 2]
        return terms
    
    def _highlight_terms(self, content: str, query_terms: List[str]) -> str:
        """Add HTML highlighting to search terms in content."""
        highlighted_content = content
        
        for term in query_terms:
            # Case-insensitive replacement with highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_content = pattern.sub(
                lambda m: f"<mark>{m.group(0)}</mark>",
                highlighted_content
            )
        
        return highlighted_content
    
    def _find_matched_terms(self, content: str, query_terms: List[str]) -> List[str]:
        """Find which query terms actually appear in the content."""
        content_lower = content.lower()
        matched_terms = []
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Check for exact term match
            if term_lower in content_lower:
                matched_terms.append(term)
            else:
                # Check for partial matches (individual words)
                for word in term_lower.split():
                    if word in content_lower and word not in matched_terms:
                        matched_terms.append(word)
        
        return matched_terms
    
    def generate_summary(self, results: List[RankedResult], query: str) -> Dict[str, Any]:
        """Generate a summary of the processed results."""
        if not results:
            return {}
        
        # Collect all quotes
        all_quotes = []
        for result in results:
            quotes_data = result.relevance_explanation.get('key_quotes', [])
            all_quotes.extend(quotes_data)
        
        # Collect all matched terms
        all_matched_terms = []
        for result in results:
            all_matched_terms.extend(result.matched_terms)
        
        # Count matches
        from collections import Counter
        term_counts = Counter(all_matched_terms)
        
        return {
            "total_results": len(results),
            "top_quotes": all_quotes[:5],  # Top 5 quotes
            "most_common_terms": dict(term_counts.most_common(10)),
            "average_relevance": sum(r.final_score for r in results) / len(results),
            "results_with_quotes": sum(1 for r in results if r.relevance_explanation.get('key_quotes')),
            "unique_sources": len(set(r.document_id for r in results))
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get post-processing service statistics."""
        return {
            "snippet_max_length": self.snippet_generator.max_snippet_length,
            "quote_patterns_count": len(self.quote_extractor.quote_patterns),
            "supported_features": [
                "snippet_generation",
                "quote_extraction", 
                "term_highlighting",
                "matched_terms_detection"
            ]
        }


# Global instance
post_processing_service = PostProcessingService()
