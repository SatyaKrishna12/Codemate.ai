"""
Query Analysis Component

Analyzes incoming queries to determine type, extract entities,
assess complexity, and optionally expand with related terms.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Optional imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

from .query_models import Query, QueryType, ComplexityLevel, Entity

# Use simple logging if app.core.logging not available
try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes queries to understand intent, extract entities, and determine complexity.
    Uses lightweight NLP techniques and pattern matching.
    """
    
    def __init__(self):
        """Initialize the query analyzer."""
        self.nlp = None
        self._init_nlp()
        self._init_patterns()
        logger.info("QueryAnalyzer initialized")
    
    def _init_nlp(self):
        """Initialize spaCy NLP pipeline (if available)."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, using pattern-based analysis only")
            self.nlp = None
            return
            
        try:
            # Try to load a small English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not available, using pattern-based analysis only")
            self.nlp = None
    
    def _init_patterns(self):
        """Initialize query type detection patterns."""
        self.query_patterns = {
            QueryType.COMPARATIVE: [
                r'\b(compare|contrast|versus|vs|difference|differ)\b',
                r'\b(better|worse|superior|inferior)\b',
                r'\b(which is|what is the difference)\b',
                r'\b(advantages|disadvantages)\b'
            ],
            QueryType.FACTUAL: [
                r'\b(what is|who is|when did|where is|how many)\b',
                r'\b(define|definition|meaning)\b',
                r'\b(fact|facts about)\b'
            ],
            QueryType.ANALYTICAL: [
                r'\b(analyze|analysis|examine|evaluate)\b',
                r'\b(why|how|what causes|what leads to)\b',
                r'\b(implications|impact|effects)\b',
                r'\b(trends|patterns)\b'
            ],
            QueryType.SUMMARIZATION: [
                r'\b(summarize|summary|overview)\b',
                r'\b(brief|outline|key points)\b',
                r'\b(in summary|tell me about)\b'
            ],
            QueryType.EXPLANATORY: [
                r'\b(explain|explanation|how does|how do)\b',
                r'\b(relationship|connection|link)\b',
                r'\b(works|functions|operates)\b'
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            'simple': [
                r'\b(what is|who is|when|where)\b',
                r'^\w+\?$'  # Single word questions
            ],
            'moderate': [
                r'\b(how|why)\b',
                r'\b(compare|contrast)\b',
                r'\band\b.*\band\b'  # Multiple entities
            ],
            'complex': [
                r'\b(analyze|evaluate|implications)\b',
                r'\b(multiple|several|various)\b',
                r'[,;].*[,;]'  # Multiple clauses
            ],
            'very_complex': [
                r'\b(comprehensive|detailed|thorough)\b',
                r'\b(step by step|process|methodology)\b',
                r'[,;].*[,;].*[,;]'  # Many clauses
            ]
        }
    
    async def analyze_query(self, query_text: str) -> Query:
        """
        Analyze a query and return a Query object with extracted information.
        
        Args:
            query_text: The original query text
            
        Returns:
            Query object with analysis results
        """
        try:
            # Create base query object
            query = Query(original_text=query_text)
            
            # Clean and process text
            query.processed_text = self._clean_text(query_text)
            
            # Detect query type
            query.query_type = self._detect_query_type(query.processed_text)
            
            # Extract entities and topics
            if self.nlp:
                query.entities = self._extract_entities_spacy(query.processed_text)
                query.topics = self._extract_topics_spacy(query.processed_text)
            else:
                query.entities = self._extract_entities_pattern(query.processed_text)
                query.topics = self._extract_topics_pattern(query.processed_text)
            
            # Determine complexity
            query.complexity = self._assess_complexity(query.processed_text)
            
            # Generate expanded terms
            query.expanded_terms = self._expand_query_terms(query.processed_text, query.entities)
            
            # Add metadata
            query.metadata = {
                'word_count': len(query.processed_text.split()),
                'char_count': len(query.processed_text),
                'has_question_mark': '?' in query_text,
                'entity_count': len(query.entities),
                'topic_count': len(query.topics)
            }
            
            logger.info(f"Analyzed query: type={query.query_type.value}, complexity={query.complexity.value}")
            return query
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return basic query on error
            return Query(
                original_text=query_text,
                processed_text=query_text,
                query_type=QueryType.UNKNOWN,
                complexity=ComplexityLevel.SIMPLE
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize query text."""
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\?\!\.,-]', '', text)  # Remove special chars
        return text
    
    def _detect_query_type(self, text: str) -> QueryType:
        """Detect the type of query based on patterns."""
        text_lower = text.lower()
        
        # Score each query type
        type_scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score += 1
            type_scores[query_type] = score
        
        # Return the type with highest score
        if type_scores and max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        return QueryType.UNKNOWN
    
    def _extract_entities_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy."""
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append(Entity(
                    text=ent.text,
                    entity_type=ent.label_,
                    confidence=0.8  # spaCy doesn't provide confidence directly
                ))
            
            return entities
        except Exception as e:
            logger.warning(f"spaCy entity extraction failed: {e}")
            return self._extract_entities_pattern(text)
    
    def _extract_entities_pattern(self, text: str) -> List[Entity]:
        """Extract entities using simple patterns."""
        entities = []
        
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in capitalized:
            entities.append(Entity(
                text=word,
                entity_type="PROPER_NOUN",
                confidence=0.6
            ))
        
        # Numbers
        numbers = re.findall(r'\b\d+\b', text)
        for num in numbers:
            entities.append(Entity(
                text=num,
                entity_type="NUMBER",
                confidence=0.9
            ))
        
        return entities
    
    def _extract_topics_spacy(self, text: str) -> List[str]:
        """Extract topic keywords using spaCy."""
        if not self.nlp:
            return self._extract_topics_pattern(text)
            
        try:
            doc = self.nlp(text)
            topics = []
            
            # Extract nouns and adjectives as topics
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    topics.append(token.lemma_.lower())
            
            return list(set(topics))  # Remove duplicates
        except Exception as e:
            logger.warning(f"spaCy topic extraction failed: {e}")
            return self._extract_topics_pattern(text)
    
    def _extract_topics_pattern(self, text: str) -> List[str]:
        """Extract topics using simple patterns."""
        # Remove common stop words and extract meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'who', 'when', 'where', 'why', 'how'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        topics = [word for word in words if word not in stop_words]
        
        return list(set(topics))
    
    def _assess_complexity(self, text: str) -> ComplexityLevel:
        """Assess query complexity based on various indicators."""
        text_lower = text.lower()
        scores = {'simple': 0, 'moderate': 0, 'complex': 0, 'very_complex': 0}
        
        # Pattern-based scoring
        for level, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    scores[level] += 1
        
        # Length-based scoring
        word_count = len(text.split())
        if word_count <= 5:
            scores['simple'] += 2
        elif word_count <= 10:
            scores['moderate'] += 1
        elif word_count <= 20:
            scores['complex'] += 1
        else:
            scores['very_complex'] += 2
        
        # Return the complexity level with highest score
        max_level = max(scores, key=scores.get)
        complexity_map = {
            'simple': ComplexityLevel.SIMPLE,
            'moderate': ComplexityLevel.MODERATE,
            'complex': ComplexityLevel.COMPLEX,
            'very_complex': ComplexityLevel.VERY_COMPLEX
        }
        
        return complexity_map[max_level]
    
    def _expand_query_terms(self, text: str, entities: List[Entity]) -> List[str]:
        """Generate expanded terms to improve query coverage."""
        expanded = []
        
        # Simple synonym expansion (can be enhanced with word embeddings)
        synonym_map = {
            'compare': ['contrast', 'evaluate', 'analyze'],
            'explain': ['describe', 'clarify', 'elaborate'],
            'find': ['search', 'locate', 'discover'],
            'show': ['display', 'demonstrate', 'illustrate'],
            'analyze': ['examine', 'study', 'investigate']
        }
        
        words = text.lower().split()
        for word in words:
            if word in synonym_map:
                expanded.extend(synonym_map[word])
        
        # Add entity variants
        for entity in entities:
            if entity.entity_type == "PROPER_NOUN":
                # Could add variations, abbreviations, etc.
                expanded.append(entity.text.lower())
        
        return list(set(expanded))
    
    def get_analysis_summary(self, query: Query) -> Dict[str, Any]:
        """Get a summary of the query analysis."""
        return {
            'query_id': query.id,
            'original_text': query.original_text,
            'query_type': query.query_type.value,
            'complexity': query.complexity.value,
            'entity_count': len(query.entities),
            'entities': [asdict(entity) for entity in query.entities],
            'topics': query.topics,
            'expanded_terms': query.expanded_terms,
            'metadata': query.metadata
        }
