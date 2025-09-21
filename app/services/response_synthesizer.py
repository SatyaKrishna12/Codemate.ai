"""
Response synthesis service using LangChain for generating answers from retrieved documents.
Implements RAG (Retrieval-Augmented Generation) for comprehensive research responses.
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import LLMError, SearchError, ValidationError
from app.models.schemas import (
    ResearchRequest, ResearchResponse, SearchResult, SourceReference, SearchRequest, SearchType
)
from app.services.query_processor import query_processor

logger = get_logger(__name__)


class ResponseSynthesizer:
    """Service for synthesizing research responses from retrieved documents."""
    
    def __init__(self):
        """Initialize the response synthesizer."""
        self._synthesis_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_context_length": 0.0
        }
        
        # Pre-defined prompts for different research scenarios
        self._prompts = {
            "default": self._get_default_prompt(),
            "summary": self._get_summary_prompt(),
            "analysis": self._get_analysis_prompt(),
            "comparison": self._get_comparison_prompt()
        }
    
    async def initialize(self) -> None:
        """Initialize the response synthesizer and its dependencies."""
        try:
            logger.info("Initializing response synthesizer")
            
            # Initialize query processor if not already done
            await query_processor.initialize()
            
            # Check if OpenAI API key is available
            if not settings.openai_api_key:
                logger.warning(
                    "OpenAI API key not configured. Response synthesis will use rule-based approach."
                )
            
            logger.info("Response synthesizer initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize response synthesizer",
                error=str(e),
                exc_info=True
            )
            raise LLMError(f"Failed to initialize response synthesizer: {str(e)}")
    
    async def research(self, request: ResearchRequest) -> ResearchResponse:
        """
        Process a research request and generate a comprehensive answer.
        
        Args:
            request: Research request with question and parameters
            
        Returns:
            Research response with answer and sources
            
        Raises:
            LLMError: If response synthesis fails
            SearchError: If document retrieval fails
            ValidationError: If request validation fails
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Processing research request",
                question=request.question[:100],  # Log first 100 chars
                context_limit=request.context_limit,
                include_sources=request.include_sources
            )
            
            # Validate request
            await self._validate_research_request(request)
            
            # Retrieve relevant documents
            search_results = await self._retrieve_context(request)
            
            # Synthesize response
            answer, confidence_score = await self._synthesize_answer(
                request.question,
                search_results
            )
            
            # Create source references
            sources = []
            if request.include_sources:
                sources = self._create_source_references(search_results)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            await self._update_synthesis_stats(processing_time, len(search_results), success=True)
            
            # Create response
            response = ResearchResponse(
                question=request.question,
                answer=answer,
                sources=sources,
                processing_time=processing_time,
                confidence_score=confidence_score,
                metadata={
                    "context_documents": len(search_results),
                    "synthesis_method": "llm" if settings.openai_api_key else "rule_based",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(
                "Research completed successfully",
                question=request.question[:100],
                answer_length=len(answer),
                sources_count=len(sources),
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
            return response
            
        except (LLMError, SearchError, ValidationError):
            await self._update_synthesis_stats(0, 0, success=False)
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            await self._update_synthesis_stats(processing_time, 0, success=False)
            logger.error(
                "Unexpected error during research",
                question=request.question[:100],
                processing_time=processing_time,
                error=str(e),
                exc_info=True
            )
            raise LLMError(
                f"Research failed: {str(e)}",
                details={"question": request.question}
            )
    
    async def _validate_research_request(self, request: ResearchRequest) -> None:
        """
        Validate research request parameters.
        
        Args:
            request: Research request to validate
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check question length
            if len(request.question.strip()) == 0:
                raise ValidationError("Question cannot be empty")
            
            if len(request.question) > 2000:
                raise ValidationError("Question is too long (maximum 2000 characters)")
            
            # Check context limit
            if not 1 <= request.context_limit <= 20:
                raise ValidationError("Context limit must be between 1 and 20")
            
            logger.debug(
                "Research request validation passed",
                question_length=len(request.question),
                context_limit=request.context_limit
            )
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Error validating research request",
                error=str(e),
                exc_info=True
            )
            raise ValidationError(f"Request validation failed: {str(e)}")
    
    async def _retrieve_context(self, request: ResearchRequest) -> List[SearchResult]:
        """
        Retrieve relevant documents for the research question.
        
        Args:
            request: Research request
            
        Returns:
            List of relevant search results
        """
        try:
            # Create search request
            search_request = SearchRequest(
                query=request.question,
                search_type=SearchType.SEMANTIC,  # Use semantic search for research
                max_results=request.context_limit,
                similarity_threshold=0.6,  # Lower threshold for research
                document_ids=request.document_ids
            )
            
            # Perform search
            search_response = await query_processor.search(search_request)
            
            logger.debug(
                "Context retrieval completed",
                question=request.question[:100],
                results_count=len(search_response.results),
                processing_time=search_response.processing_time
            )
            
            return search_response.results
            
        except Exception as e:
            logger.error(
                "Error retrieving context",
                question=request.question[:100],
                error=str(e),
                exc_info=True
            )
            raise SearchError(f"Failed to retrieve context: {str(e)}")
    
    async def _synthesize_answer(
        self,
        question: str,
        search_results: List[SearchResult]
    ) -> Tuple[str, Optional[float]]:
        """
        Synthesize an answer from the retrieved documents.
        
        Args:
            question: Research question
            search_results: Retrieved document chunks
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        try:
            if not search_results:
                return self._create_no_context_answer(question), 0.1
            
            # Check if OpenAI API is available
            if settings.openai_api_key:
                return await self._llm_synthesize(question, search_results)
            else:
                return await self._rule_based_synthesize(question, search_results)
            
        except Exception as e:
            logger.error(
                "Error synthesizing answer",
                question=question[:100],
                context_count=len(search_results),
                error=str(e),
                exc_info=True
            )
            raise LLMError(f"Failed to synthesize answer: {str(e)}")
    
    async def _llm_synthesize(
        self,
        question: str,
        search_results: List[SearchResult]
    ) -> Tuple[str, Optional[float]]:
        """
        Synthesize answer using LLM (OpenAI GPT).
        
        Args:
            question: Research question
            search_results: Retrieved document chunks
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        try:
            # This is a placeholder for LangChain LLM integration
            # In a production system, you would use LangChain's OpenAI integration
            logger.warning("LLM synthesis not fully implemented, falling back to rule-based approach")
            
            return await self._rule_based_synthesize(question, search_results)
            
        except Exception as e:
            logger.error(
                "Error in LLM synthesis",
                error=str(e),
                exc_info=True
            )
            # Fallback to rule-based approach
            return await self._rule_based_synthesize(question, search_results)
    
    async def _rule_based_synthesize(
        self,
        question: str,
        search_results: List[SearchResult]
    ) -> Tuple[str, Optional[float]]:
        """
        Synthesize answer using rule-based approach.
        
        Args:
            question: Research question
            search_results: Retrieved document chunks
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        try:
            logger.debug(
                "Using rule-based synthesis",
                question=question[:100],
                context_count=len(search_results)
            )
            
            # Sort results by relevance score
            sorted_results = sorted(
                search_results,
                key=lambda x: x.similarity_score,
                reverse=True
            )
            
            # Extract key information
            relevant_content = []
            total_relevance = 0
            
            for result in sorted_results[:5]:  # Use top 5 results
                content = result.content.strip()
                if content:
                    relevant_content.append(f"• {content}")
                    total_relevance += result.similarity_score
            
            # Calculate confidence based on relevance scores
            confidence_score = None
            if sorted_results:
                average_relevance = total_relevance / len(sorted_results[:5])
                confidence_score = min(0.9, average_relevance)  # Cap at 0.9 for rule-based
            
            # Create answer
            if relevant_content:
                answer = self._format_rule_based_answer(question, relevant_content, sorted_results)
            else:
                answer = self._create_no_context_answer(question)
                confidence_score = 0.1
            
            logger.debug(
                "Rule-based synthesis completed",
                answer_length=len(answer),
                confidence_score=confidence_score
            )
            
            return answer, confidence_score
            
        except Exception as e:
            logger.error(
                "Error in rule-based synthesis",
                error=str(e),
                exc_info=True
            )
            return self._create_no_context_answer(question), 0.1
    
    def _format_rule_based_answer(
        self,
        question: str,
        relevant_content: List[str],
        search_results: List[SearchResult]
    ) -> str:
        """Format answer using rule-based approach."""
        try:
            # Determine question type for better formatting
            question_lower = question.lower()
            
            if any(word in question_lower for word in ["what", "who", "where", "when"]):
                intro = "Based on the available documents, here's what I found:"
            elif any(word in question_lower for word in ["how", "why"]):
                intro = "Based on the available documents, here's an explanation:"
            elif any(word in question_lower for word in ["compare", "difference", "versus"]):
                intro = "Based on the available documents, here's a comparison:"
            else:
                intro = "Based on the available documents:"
            
            # Build answer
            answer_parts = [intro, ""]
            answer_parts.extend(relevant_content[:3])  # Limit to top 3 points
            
            # Add summary if multiple sources
            if len(search_results) > 1:
                answer_parts.extend([
                    "",
                    f"This information is compiled from {len(search_results)} relevant document sections."
                ])
            
            return "\n".join(answer_parts)
            
        except Exception as e:
            logger.error(
                "Error formatting rule-based answer",
                error=str(e),
                exc_info=True
            )
            return "I found some relevant information but encountered an error while formatting the response."
    
    def _create_no_context_answer(self, question: str) -> str:
        """Create answer when no relevant context is found."""
        return (
            f"I couldn't find specific information to answer your question: '{question[:100]}...'\n\n"
            "This might be because:\n"
            "• The question relates to information not present in the uploaded documents\n"
            "• The query terms don't match the content in the knowledge base\n"
            "• You might need to rephrase your question or upload more relevant documents\n\n"
            "Try rephrasing your question or adding more specific keywords."
        )
    
    def _create_source_references(self, search_results: List[SearchResult]) -> List[SourceReference]:
        """Create source references from search results."""
        try:
            sources = []
            
            for result in search_results:
                source = SourceReference(
                    document_id=result.document_id,
                    document_filename=result.document_filename,
                    chunk_id=result.chunk_id,
                    chunk_index=result.chunk_index,
                    relevance_score=result.similarity_score,
                    excerpt=result.content[:200] + "..." if len(result.content) > 200 else result.content
                )
                sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(
                "Error creating source references",
                error=str(e),
                exc_info=True
            )
            return []
    
    async def _update_synthesis_stats(
        self,
        processing_time: float,
        context_count: int,
        success: bool
    ) -> None:
        """Update synthesis statistics."""
        try:
            self._synthesis_stats["total_requests"] += 1
            self._synthesis_stats["total_processing_time"] += processing_time
            
            if success:
                self._synthesis_stats["successful_requests"] += 1
                
                # Update average context length
                current_avg = self._synthesis_stats["average_context_length"]
                total_requests = self._synthesis_stats["successful_requests"]
                self._synthesis_stats["average_context_length"] = (
                    (current_avg * (total_requests - 1) + context_count) / total_requests
                )
            else:
                self._synthesis_stats["failed_requests"] += 1
            
        except Exception as e:
            logger.warning(
                "Error updating synthesis statistics",
                error=str(e)
            )
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template."""
        return """
        You are a helpful research assistant. Based on the provided context documents, 
        answer the user's question comprehensively and accurately.
        
        Context: {context}
        
        Question: {question}
        
        Instructions:
        - Provide a clear, well-structured answer based on the context
        - If the context doesn't contain enough information, state this clearly
        - Cite specific sources when possible
        - Be concise but thorough
        
        Answer:
        """
    
    def _get_summary_prompt(self) -> str:
        """Get summary prompt template."""
        return """
        Provide a comprehensive summary based on the following documents.
        
        Context: {context}
        
        Create a summary that captures the main points, key findings, and important details.
        """
    
    def _get_analysis_prompt(self) -> str:
        """Get analysis prompt template."""
        return """
        Analyze the provided information and answer the question with detailed reasoning.
        
        Context: {context}
        
        Question: {question}
        
        Provide an analytical response that explains the reasoning behind your conclusions.
        """
    
    def _get_comparison_prompt(self) -> str:
        """Get comparison prompt template."""
        return """
        Compare and contrast the information in the provided context to answer the question.
        
        Context: {context}
        
        Question: {question}
        
        Highlight similarities, differences, and provide a balanced comparison.
        """
    
    async def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        try:
            stats = self._synthesis_stats.copy()
            
            # Calculate success rate
            if stats["total_requests"] > 0:
                stats["success_rate"] = (
                    stats["successful_requests"] / stats["total_requests"]
                )
                stats["average_processing_time"] = (
                    stats["total_processing_time"] / stats["total_requests"]
                )
            else:
                stats["success_rate"] = 0.0
                stats["average_processing_time"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(
                "Error getting synthesis statistics",
                error=str(e),
                exc_info=True
            )
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the response synthesizer."""
        try:
            # Check query processor dependency
            query_processor_health = await query_processor.health_check()
            
            dependencies_healthy = query_processor_health.get("status") == "healthy"
            
            status = {
                "service": "response_synthesizer",
                "status": "healthy" if dependencies_healthy else "unhealthy",
                "llm_available": bool(settings.openai_api_key),
                "synthesis_method": "llm" if settings.openai_api_key else "rule_based",
                "dependencies": {
                    "query_processor": query_processor_health
                },
                "synthesis_stats": await self.get_synthesis_stats()
            }
            
            return status
            
        except Exception as e:
            logger.error(
                "Response synthesizer health check failed",
                error=str(e),
                exc_info=True
            )
            return {
                "service": "response_synthesizer",
                "status": "unhealthy",
                "error": str(e)
            }


# Global response synthesizer instance
response_synthesizer = ResponseSynthesizer()
