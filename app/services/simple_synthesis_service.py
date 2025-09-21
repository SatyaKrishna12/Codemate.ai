"""
Simplified Synthesis Service using direct Groq API calls.

This provides the core synthesis functionality without complex LangChain dependencies.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

from groq import Groq
from dotenv import load_dotenv

from app.core.config import settings
from app.core.logging import get_logger
from app.models.synthesis_schemas import (
    SynthesisRequest, SynthesisResponse, SynthesisConfig, OutputFormat, 
    OutputStyle, CitationStyle, ContentSection, SynthesisMetrics
)

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class SimpleSynthesisService:
    """Simplified synthesis service using direct Groq API."""
    
    def __init__(self):
        """Initialize the synthesis service."""
        self.groq_client = None
        self.model_name = "llama-3.1-8b-instant"
        self._initialize_groq()
    
    def _initialize_groq(self):
        """Initialize Groq client."""
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logger.warning("GROQ_API_KEY not found in environment")
                return
            
            self.groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
    
    async def generate_synthesis(self, request: SynthesisRequest) -> SynthesisResponse:
        """Generate a synthesis response."""
        start_time = time.time()
        
        try:
            if not self.groq_client:
                raise Exception("Groq client not initialized")
            
            # Prepare context from sources
            context = self._prepare_context(request.sources)
            
            # Create prompt based on output format
            prompt = self._create_prompt(request.query, context, request.config)
            
            # Generate response using Groq
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst and content synthesizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Format the content for better presentation
            formatted_content = self._format_response(content)
            
            # Process the response
            sections = self._process_response(formatted_content, request.config.output_format)
            
            # Create metrics
            processing_time = time.time() - start_time
            metrics = SynthesisMetrics(
                processing_time=processing_time,
                tokens_used=response.usage.total_tokens,
                sources_processed=len(request.sources),
                assertions_extracted=len(sections),
                confidence_score=0.85  # Simplified confidence
            )
            
            return SynthesisResponse(
                synthesis_id=f"synthesis_{int(time.time())}",
                query=request.query,
                content=formatted_content,
                sections=sections,
                citations=[],  # Simplified - no complex citation processing
                highlights=[],
                metrics=metrics,
                metadata={
                    "model": self.model_name,
                    "format": request.config.output_format.value,
                    "style": request.config.output_style.value,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in synthesis generation: {e}")
            raise
    
    def _prepare_context(self, sources: List[Dict[str, Any]]) -> str:
        """Prepare context from sources."""
        context_parts = []
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown')
            content = source.get('content', source.get('text', ''))
            context_parts.append(f"[Source-{i}] {title}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str, config: SynthesisConfig) -> str:
        """Create synthesis prompt based on configuration with improved formatting."""
        base_prompt = f"""Using ONLY the provided context documents, synthesize a comprehensive response to the user's question.

INSTRUCTIONS:
- Use only information from the provided context
- Include citations using [Source-X] format
- Structure response according to format: {config.output_format.value}
- Style: {config.output_style.value}

FORMATTING REQUIREMENTS:
- Use **bold text** for important points and headings (not asterisks)
- Use bullet points (•) for lists instead of asterisks (*)
- Use numbered lists (1., 2., 3.) for sequential information
- Use clear paragraph breaks for better readability
- Ensure professional presentation

USER QUESTION: {query}

CONTEXT:
{context}

RESPONSE:"""
        
        return base_prompt
    
    def _process_response(self, content: str, output_format: OutputFormat) -> List[ContentSection]:
        """Process response into sections."""
        # Simplified section processing
        sections = []
        
        if output_format == OutputFormat.EXECUTIVE_SUMMARY:
            sections.append(ContentSection(
                title="Executive Summary",
                content=content,
                citations=[],
                confidence_score=0.85
            ))
        elif output_format == OutputFormat.DETAILED_REPORT:
            # Split content into sections (simplified)
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if para.strip():
                    sections.append(ContentSection(
                        title=f"Section {i+1}",
                        content=para.strip(),
                        citations=[],
                        confidence_score=0.85
                    ))
        else:
            sections.append(ContentSection(
                title="Response",
                content=content,
                citations=[],
                confidence_score=0.85
            ))
        
        return sections
    
    def _format_response(self, content: str) -> str:
        """Post-process response to ensure proper formatting."""
        if not content:
            return content
            
        # Replace asterisk formatting with proper markdown bold
        import re
        
        # Replace **text** patterns (already correct)
        # Replace *text* patterns with **text**
        content = re.sub(r'\*([^*]+)\*', r'**\1**', content)
        
        # Replace asterisk bullet points with proper bullet points
        content = re.sub(r'^\s*\*\s+', '• ', content, flags=re.MULTILINE)
        
        # Replace multiple asterisks at start of line with bullet points
        content = re.sub(r'^\s*\*{2,}\s*', '• ', content, flags=re.MULTILINE)
        
        # Ensure proper spacing around headings
        content = re.sub(r'\n(\*\*[^*]+\*\*)\n', r'\n\n\1\n\n', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()

    async def quick_synthesis(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Quick synthesis for immediate responses with improved formatting."""
        try:
            if not self.groq_client:
                return "Synthesis service not available"
            
            context = self._prepare_context(sources)
            
            # Enhanced prompt for better formatting
            prompt = f"""Based on the provided context, answer the following question with proper formatting:

QUESTION: {query}

FORMATTING REQUIREMENTS:
- Use **bold text** for important points and headings (not asterisks)
- Use bullet points (•) for lists instead of asterisks (*)
- Use numbered lists (1., 2., 3.) for sequential information
- Use clear paragraph breaks for better readability
- Include [Source-X] citations where relevant

CONTEXT:
{context}

Please provide a well-formatted, professional response:"""
            
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,  # Increased for better formatting
                temperature=0.3  # Lower temperature for more consistent formatting
            )
            
            # Format the response for better presentation
            raw_content = response.choices[0].message.content
            formatted_content = self._format_response(raw_content)
            
            return formatted_content
            
        except Exception as e:
            logger.error(f"Error in quick synthesis: {e}")
            return f"Error generating synthesis: {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        return {
            "service": "SimpleSynthesisService",
            "status": "healthy" if self.groq_client else "degraded",
            "groq_available": self.groq_client is not None,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat()
        }


# Global instance
simple_synthesis_service = SimpleSynthesisService()
