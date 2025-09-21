"""
Query Decomposition Component

Breaks down complex queries into manageable sub-questions
while maintaining logical relationships and dependencies.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from .query_models import Query, SubQuery, QueryType, ComplexityLevel
from app.core.logging import get_logger

logger = get_logger(__name__)


class QueryDecomposer:
    """
    Decomposes complex queries into sub-questions with logical relationships.
    Maintains dependencies and execution order for multi-step reasoning.
    """
    
    def __init__(self):
        """Initialize the query decomposer."""
        self._init_decomposition_patterns()
        logger.info("QueryDecomposer initialized")
    
    def _init_decomposition_patterns(self):
        """Initialize patterns for different decomposition strategies."""
        
        # Patterns for comparative queries
        self.comparative_patterns = {
            'explicit_comparison': r'\b(compare|contrast)\s+(.+?)\s+(and|with|to|versus|vs)\s+(.+?)(?:\?|$)',
            'vs_pattern': r'(.+?)\s+(vs|versus)\s+(.+?)(?:\?|$)',
            'difference_pattern': r'(?:what is|what are)\s+(?:the\s+)?(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)',
            'which_better': r'which\s+(?:is\s+)?(?:better|superior|worse|inferior)[\s:]+(.+?)\s+(?:or|vs|versus)\s+(.+?)(?:\?|$)'
        }
        
        # Patterns for analytical queries
        self.analytical_patterns = {
            'cause_effect': r'(?:why|what causes|what leads to|how does)\s+(.+?)(?:\?|$)',
            'process_explanation': r'(?:how|explain how)\s+(.+?)\s+(?:works?|functions?|operates?)(?:\?|$)',
            'analysis_request': r'(?:analyze|examine|evaluate)\s+(.+?)(?:\?|$)'
        }
        
        # Patterns for multi-part queries
        self.multi_part_patterns = {
            'and_connector': r'(.+?)\s+and\s+(.+?)(?:\s+and\s+(.+?))?(?:\?|$)',
            'sequential': r'(?:first|then|next|finally|also)',
            'bullet_points': r'[•\-\*]\s*(.+?)(?:\n|$)'
        }
    
    async def decompose_query(self, query: Query) -> List[SubQuery]:
        """
        Decompose a query into sub-questions based on its type and complexity.
        
        Args:
            query: The analyzed query object
            
        Returns:
            List of SubQuery objects with dependencies
        """
        try:
            sub_queries = []
            
            # Skip decomposition for simple queries
            if query.complexity == ComplexityLevel.SIMPLE:
                logger.info(f"Query {query.id} is simple, no decomposition needed")
                return sub_queries
            
            # Decompose based on query type
            if query.query_type == QueryType.COMPARATIVE:
                sub_queries = self._decompose_comparative(query)
            elif query.query_type == QueryType.ANALYTICAL:
                sub_queries = self._decompose_analytical(query)
            elif query.query_type == QueryType.EXPLANATORY:
                sub_queries = self._decompose_explanatory(query)
            elif query.query_type == QueryType.SUMMARIZATION:
                sub_queries = self._decompose_summarization(query)
            else:
                # Generic decomposition for unknown types
                sub_queries = self._decompose_generic(query)
            
            # Set dependencies and order
            sub_queries = self._establish_dependencies(sub_queries, query)
            
            logger.info(f"Decomposed query {query.id} into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error decomposing query {query.id}: {e}")
            return []
    
    def _decompose_comparative(self, query: Query) -> List[SubQuery]:
        """Decompose comparative queries."""
        sub_queries = []
        text = query.processed_text
        
        # Try to find entities being compared
        entities = self._extract_comparison_entities(text)
        
        if len(entities) >= 2:
            # Create sub-queries to gather information about each entity
            for i, entity in enumerate(entities):
                sub_query = SubQuery(
                    text=f"What are the key characteristics and features of {entity}?",
                    parent_query_id=query.id,
                    order=i + 1,
                    metadata={'entity': entity, 'purpose': 'information_gathering'}
                )
                sub_queries.append(sub_query)
            
            # Create comparison sub-query
            comparison_text = f"Compare and contrast {' and '.join(entities)} based on their characteristics."
            comparison_query = SubQuery(
                text=comparison_text,
                parent_query_id=query.id,
                order=len(entities) + 1,
                dependencies=[sq.id for sq in sub_queries],  # Depends on all info-gathering queries
                metadata={'purpose': 'comparison'}
            )
            sub_queries.append(comparison_query)
        
        return sub_queries
    
    def _decompose_analytical(self, query: Query) -> List[SubQuery]:
        """Decompose analytical queries."""
        sub_queries = []
        text = query.processed_text.lower()
        
        # For cause-effect analysis
        if any(word in text for word in ['why', 'cause', 'reason', 'leads to']):
            # First gather background information
            background_query = SubQuery(
                text=f"What is the background and context for: {query.processed_text}",
                parent_query_id=query.id,
                order=1,
                metadata={'purpose': 'background'}
            )
            sub_queries.append(background_query)
            
            # Then identify causes/factors
            causes_query = SubQuery(
                text=f"What are the main factors and causes related to: {query.processed_text}",
                parent_query_id=query.id,
                order=2,
                dependencies=[background_query.id],
                metadata={'purpose': 'cause_identification'}
            )
            sub_queries.append(causes_query)
            
            # Finally analyze relationships
            analysis_query = SubQuery(
                text=f"How do these factors interact and what are the implications?",
                parent_query_id=query.id,
                order=3,
                dependencies=[causes_query.id],
                metadata={'purpose': 'relationship_analysis'}
            )
            sub_queries.append(analysis_query)
        
        return sub_queries
    
    def _decompose_explanatory(self, query: Query) -> List[SubQuery]:
        """Decompose explanatory queries."""
        sub_queries = []
        text = query.processed_text
        
        # Extract the main concept to explain
        concept = self._extract_main_concept(text)
        
        if concept:
            # Basic definition/overview
            definition_query = SubQuery(
                text=f"What is {concept} and what are its basic characteristics?",
                parent_query_id=query.id,
                order=1,
                metadata={'concept': concept, 'purpose': 'definition'}
            )
            sub_queries.append(definition_query)
            
            # How it works/functions
            if any(word in text.lower() for word in ['how', 'works', 'functions', 'operates']):
                mechanism_query = SubQuery(
                    text=f"How does {concept} work or function?",
                    parent_query_id=query.id,
                    order=2,
                    dependencies=[definition_query.id],
                    metadata={'concept': concept, 'purpose': 'mechanism'}
                )
                sub_queries.append(mechanism_query)
            
            # Examples and applications
            examples_query = SubQuery(
                text=f"What are some examples and real-world applications of {concept}?",
                parent_query_id=query.id,
                order=3,
                dependencies=[definition_query.id],
                metadata={'concept': concept, 'purpose': 'examples'}
            )
            sub_queries.append(examples_query)
        
        return sub_queries
    
    def _decompose_summarization(self, query: Query) -> List[SubQuery]:
        """Decompose summarization queries."""
        sub_queries = []
        text = query.processed_text
        
        # Extract topic to summarize
        topic = self._extract_summarization_topic(text)
        
        if topic:
            # Key points
            key_points_query = SubQuery(
                text=f"What are the key points and main aspects of {topic}?",
                parent_query_id=query.id,
                order=1,
                metadata={'topic': topic, 'purpose': 'key_points'}
            )
            sub_queries.append(key_points_query)
            
            # Important details
            details_query = SubQuery(
                text=f"What are the important details and specifics about {topic}?",
                parent_query_id=query.id,
                order=2,
                metadata={'topic': topic, 'purpose': 'details'}
            )
            sub_queries.append(details_query)
            
            # Synthesis
            synthesis_query = SubQuery(
                text=f"Synthesize the information to create a comprehensive summary of {topic}.",
                parent_query_id=query.id,
                order=3,
                dependencies=[key_points_query.id, details_query.id],
                metadata={'topic': topic, 'purpose': 'synthesis'}
            )
            sub_queries.append(synthesis_query)
        
        return sub_queries
    
    def _decompose_generic(self, query: Query) -> List[SubQuery]:
        """Generic decomposition for complex queries."""
        sub_queries = []
        
        # Try to split on conjunctions
        parts = self._split_on_conjunctions(query.processed_text)
        
        if len(parts) > 1:
            for i, part in enumerate(parts):
                sub_query = SubQuery(
                    text=part.strip(),
                    parent_query_id=query.id,
                    order=i + 1,
                    metadata={'purpose': 'generic_split', 'part_number': i + 1}
                )
                sub_queries.append(sub_query)
        
        return sub_queries
    
    def _extract_comparison_entities(self, text: str) -> List[str]:
        """Extract entities being compared from text."""
        entities = []
        
        # Try different comparison patterns
        for pattern_name, pattern in self.comparative_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if pattern_name == 'explicit_comparison':
                    entities = [match.group(2).strip(), match.group(4).strip()]
                elif pattern_name in ['vs_pattern', 'difference_pattern', 'which_better']:
                    entities = [match.group(1).strip(), match.group(2).strip()]
                break
        
        # Clean entities
        entities = [entity.strip('" \'') for entity in entities if entity.strip()]
        return entities
    
    def _extract_main_concept(self, text: str) -> Optional[str]:
        """Extract the main concept to be explained."""
        # Remove question words and common phrases
        concept = re.sub(r'\b(explain|how|what|why|does|do|is|are|the|a|an)\b', '', text, flags=re.IGNORECASE)
        concept = re.sub(r'\b(work|works|function|functions|operate|operates)\b', '', concept, flags=re.IGNORECASE)
        concept = concept.strip(' ?.,!')
        
        return concept if concept else None
    
    def _extract_summarization_topic(self, text: str) -> Optional[str]:
        """Extract the topic to be summarized."""
        # Remove summarization keywords
        topic = re.sub(r'\b(summarize|summary|overview|about|of|the|a|an)\b', '', text, flags=re.IGNORECASE)
        topic = topic.strip(' ?.,!')
        
        return topic if topic else None
    
    def _split_on_conjunctions(self, text: str) -> List[str]:
        """Split text on conjunctions and connectors."""
        # Split on common conjunctions
        conjunctions = r'\b(and|or|also|additionally|furthermore|moreover)\b'
        parts = re.split(conjunctions, text, flags=re.IGNORECASE)
        
        # Filter out the conjunctions themselves and empty parts
        filtered_parts = []
        for part in parts:
            cleaned = part.strip()
            if cleaned and not re.match(conjunctions, cleaned, re.IGNORECASE):
                filtered_parts.append(cleaned)
        
        return filtered_parts if len(filtered_parts) > 1 else [text]
    
    def _establish_dependencies(self, sub_queries: List[SubQuery], parent_query: Query) -> List[SubQuery]:
        """Establish dependencies and execution order for sub-queries."""
        if not sub_queries:
            return sub_queries
        
        # Sort by order
        sub_queries.sort(key=lambda sq: sq.order)
        
        # For analytical and explanatory queries, create sequential dependencies
        if parent_query.query_type in [QueryType.ANALYTICAL, QueryType.EXPLANATORY]:
            for i in range(1, len(sub_queries)):
                if not sub_queries[i].dependencies:
                    sub_queries[i].dependencies = [sub_queries[i-1].id]
        
        return sub_queries
    
    def get_decomposition_summary(self, sub_queries: List[SubQuery]) -> Dict[str, Any]:
        """Get a summary of the query decomposition."""
        return {
            'sub_query_count': len(sub_queries),
            'sub_queries': [
                {
                    'id': sq.id,
                    'text': sq.text,
                    'order': sq.order,
                    'dependencies': sq.dependencies,
                    'metadata': sq.metadata
                }
                for sq in sub_queries
            ],
            'dependency_graph': self._build_dependency_graph(sub_queries)
        }
    
    def _build_dependency_graph(self, sub_queries: List[SubQuery]) -> Dict[str, List[str]]:
        """Build a dependency graph for visualization."""
        graph = {}
        for sq in sub_queries:
            graph[sq.id] = sq.dependencies
        return graph
