"""
Intelligent Query Processing System

Main orchestrator that combines all components to provide comprehensive
query processing with analysis, decomposition, execution, and context management.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .query_models import QueryProcessingResult, ExecutionContext
from .query_analyzer import QueryAnalyzer
from .query_decomposer import QueryDecomposer
from .query_executor import QueryExecutor
from .context_manager import ContextManager

# Use simple logging if app.core.logging not available
try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class IntelligentQueryProcessor:
    """
    Main query processing system that orchestrates analysis, decomposition,
    execution, and context management for intelligent query handling.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, 
                 enable_context_management: bool = True):
        """
        Initialize the intelligent query processor.
        
        Args:
            openai_api_key: OpenAI API key for LangChain integration
            enable_context_management: Whether to enable context management
        """
        self.analyzer = QueryAnalyzer()
        self.decomposer = QueryDecomposer()
        self.executor = QueryExecutor(openai_api_key)
        
        self.context_manager = None
        if enable_context_management:
            self.context_manager = ContextManager()
        
        self.processing_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0
        }
        
        logger.info("IntelligentQueryProcessor initialized")
    
    async def process_query(self, query_text: str, session_id: Optional[str] = None,
                          user_preferences: Optional[Dict[str, Any]] = None) -> QueryProcessingResult:
        """
        Process a query through the complete pipeline.
        
        Args:
            query_text: The user's query
            session_id: Optional session ID for context management
            user_preferences: Optional user preferences
            
        Returns:
            QueryProcessingResult with comprehensive results
        """
        start_time = time.time()
        
        try:
            # Update stats
            self.processing_stats['total_queries'] += 1
            
            # Initialize session if context management is enabled
            if self.context_manager and not session_id:
                session_id = self.context_manager.create_session()
            elif self.context_manager and session_id:
                # Ensure session exists
                if not self.context_manager.get_session(session_id):
                    self.context_manager.create_session(session_id)
            
            # Get execution context
            context = self._prepare_execution_context(session_id, query_text, user_preferences)
            
            # Step 1: Analyze the query
            logger.info(f"Analyzing query: {query_text[:100]}...")
            query = await self.analyzer.analyze_query(query_text)
            
            # Check for follow-up intent if context management is enabled
            if self.context_manager and session_id:
                follow_up_analysis = self.context_manager.detect_follow_up_intent(session_id, query_text)
                if follow_up_analysis['is_follow_up']:
                    # Enhance query with relevant context
                    relevant_context = self.context_manager.get_relevant_context(session_id, query_text)
                    if relevant_context:
                        query.processed_text = f"{query.processed_text}\n\nRelevant context:\n{relevant_context}"
                        query.metadata['follow_up'] = follow_up_analysis
            
            # Step 2: Decompose the query if needed
            logger.info(f"Decomposing query (type: {query.query_type.value}, complexity: {query.complexity.value})")
            sub_queries = await self.decomposer.decompose_query(query)
            
            # Step 3: Execute the query with sub-queries
            logger.info(f"Executing query with {len(sub_queries)} sub-queries")
            result = await self.executor.execute_query(query, sub_queries, context)
            
            # Step 4: Update context if context management is enabled
            if self.context_manager and session_id:
                self.context_manager.update_session_context(session_id, query, result)
            
            # Update stats
            execution_time = (time.time() - start_time) * 1000
            self.processing_stats['successful_queries'] += 1
            self.processing_stats['total_processing_time_ms'] += execution_time
            self.processing_stats['avg_processing_time_ms'] = (
                self.processing_stats['total_processing_time_ms'] / 
                self.processing_stats['total_queries']
            )
            
            # Add processing metadata
            result.metadata.update({
                'session_id': session_id,
                'processing_time_ms': execution_time,
                'analysis_summary': self.analyzer.get_analysis_summary(query),
                'decomposition_summary': self.decomposer.get_decomposition_summary(sub_queries) if sub_queries else None
            })
            
            logger.info(f"Successfully processed query {query.id} in {execution_time:.2f}ms")
            return result
            
        except Exception as e:
            # Update error stats
            self.processing_stats['failed_queries'] += 1
            execution_time = (time.time() - start_time) * 1000
            
            logger.error(f"Error processing query: {e}")
            
            # Return error result
            return QueryProcessingResult(
                query_id="error",
                original_query=query_text,
                final_answer=f"I apologize, but I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                execution_time_ms=execution_time,
                metadata={'error': str(e), 'session_id': session_id}
            )
    
    def _prepare_execution_context(self, session_id: Optional[str], query_text: str,
                                 user_preferences: Optional[Dict[str, Any]]) -> ExecutionContext:
        """Prepare execution context for query processing."""
        if self.context_manager and session_id:
            context = self.context_manager.get_session(session_id)
            if context:
                if user_preferences:
                    context.user_preferences.update(user_preferences)
                return context
        
        # Create new context if not available
        return ExecutionContext(
            session_id=session_id or "standalone",
            user_preferences=user_preferences or {}
        )
    
    async def process_batch_queries(self, queries: List[str], 
                                  session_id: Optional[str] = None) -> List[QueryProcessingResult]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query strings
            session_id: Optional session ID for context
            
        Returns:
            List of QueryProcessingResult objects
        """
        results = []
        
        for query_text in queries:
            try:
                result = await self.process_query(query_text, session_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch query '{query_text}': {e}")
                # Add error result
                results.append(QueryProcessingResult(
                    query_id="batch_error",
                    original_query=query_text,
                    final_answer=f"Error processing query: {str(e)}",
                    confidence=0.0,
                    metadata={'batch_error': True}
                ))
        
        return results
    
    def get_session_analysis(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis of a conversation session."""
        if not self.context_manager:
            return None
        
        return self.context_manager.analyze_conversation_context(session_id)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics and performance metrics."""
        return {
            **self.processing_stats,
            'success_rate': (
                self.processing_stats['successful_queries'] / 
                max(self.processing_stats['total_queries'], 1)
            ) * 100,
            'error_rate': (
                self.processing_stats['failed_queries'] / 
                max(self.processing_stats['total_queries'], 1)
            ) * 100
        }
    
    def export_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session data for analysis or backup."""
        if not self.context_manager:
            return None
        
        return self.context_manager.export_session(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old sessions."""
        if not self.context_manager:
            return 0
        
        return self.context_manager.cleanup_old_sessions(max_age_hours)
    
    async def explain_processing_steps(self, query_text: str) -> Dict[str, Any]:
        """
        Explain how a query would be processed without actually executing it.
        Useful for debugging and understanding the system behavior.
        """
        try:
            # Analyze the query
            query = await self.analyzer.analyze_query(query_text)
            analysis_summary = self.analyzer.get_analysis_summary(query)
            
            # Decompose the query
            sub_queries = await self.decomposer.decompose_query(query)
            decomposition_summary = self.decomposer.get_decomposition_summary(sub_queries)
            
            return {
                'original_query': query_text,
                'analysis': analysis_summary,
                'decomposition': decomposition_summary,
                'execution_plan': {
                    'steps': [
                        {
                            'step': 1,
                            'action': 'Query Analysis',
                            'description': f"Identify query as {query.query_type.value} with {query.complexity.value} complexity"
                        },
                        {
                            'step': 2,
                            'action': 'Query Decomposition',
                            'description': f"Break down into {len(sub_queries)} sub-queries" if sub_queries else "No decomposition needed"
                        },
                        {
                            'step': 3,
                            'action': 'Execution',
                            'description': "Execute sub-queries in dependency order, then main query"
                        },
                        {
                            'step': 4,
                            'action': 'Context Update',
                            'description': "Update conversation context and memory"
                        }
                    ]
                },
                'estimated_complexity': query.complexity.value,
                'estimated_execution_time': self._estimate_execution_time(query, sub_queries)
            }
            
        except Exception as e:
            logger.error(f"Error explaining processing steps: {e}")
            return {'error': str(e)}
    
    def _estimate_execution_time(self, query, sub_queries) -> str:
        """Estimate execution time based on query complexity."""
        base_time = {
            'simple': 1.0,
            'moderate': 3.0,
            'complex': 8.0,
            'very_complex': 15.0
        }
        
        estimated_seconds = base_time.get(query.complexity.value, 5.0)
        estimated_seconds += len(sub_queries) * 2.0  # Additional time for sub-queries
        
        if estimated_seconds < 60:
            return f"~{estimated_seconds:.0f} seconds"
        else:
            return f"~{estimated_seconds/60:.1f} minutes"


# Global instance for easy access
intelligent_query_processor = IntelligentQueryProcessor()
