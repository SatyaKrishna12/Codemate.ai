"""
Main Synthesis Service for LLM-powered response generation and content synthesis.

This service orchestrates the entire synthesis pipeline from retrieval through
LLM generation to quality checking and formatting. It integrates with Groq LLM
via LangChain and provides structured outputs with citations and quality validation.
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import markdown

from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from app.core.config import settings
from app.core.logging import get_logger
from app.models.synthesis_schemas import (
    SynthesisRequest, SynthesisResponse, SynthesisConfig, OutputFormat, 
    OutputStyle, CitationStyle, Assertion, ContentSection, Highlight,
    SupportingSource, SourceMetadata, DryRunResult, SynthesisMetrics,
    AssertionExplanation, QualityCheck
)
from app.services.citation_manager import CitationManager, get_citation_manager
from app.services.quality_checker import quality_checker
from app.services.comprehensive_ir_system import comprehensive_ir_system

logger = get_logger(__name__)


class PromptTemplates:
    """Collection of prompt templates for different synthesis tasks."""
    
    SYNTHESIS_TEMPLATE = """You are an expert research analyst. Using ONLY the provided context documents, synthesize a comprehensive response to the user's question.

CRITICAL INSTRUCTIONS:
1. Use ONLY information from the provided context - do not add external knowledge
2. If information is insufficient, explicitly state "Not enough information available"
3. Include specific citations for ALL claims using the format [Source-X] where X is the document number
4. Maintain factual accuracy and acknowledge contradictions if they exist
5. Structure your response according to the requested format and style

USER QUESTION: {query}

REQUESTED FORMAT: {format}
REQUESTED STYLE: {style}

CONTEXT DOCUMENTS:
{context}

RESPONSE REQUIREMENTS:
- Start with a clear introduction
- Use proper headings and sections for the requested format
- Include in-text citations [Source-1], [Source-2], etc.
- End with key takeaways or conclusion
- Flag any contradictory information found
- If creating FAQ format, provide 5-8 clear question-answer pairs
- If creating bullet points, use clear, concise points with citations

Generate your response in markdown format with proper structure and citations:"""

    ASSERTION_EXTRACTION_TEMPLATE = """Analyze the following generated content and extract key factual assertions.

For each assertion:
1. Identify specific factual claims
2. Note which sources support each claim
3. Rate the consensus level (0.0-1.0)
4. Flag any conflicts between sources

CONTENT TO ANALYZE:
{content}

SOURCE DOCUMENTS:
{sources}

Return a JSON object with this structure:
{{
    "assertions": [
        {{
            "id": "assert-1",
            "text": "specific factual claim",
            "consensus_score": 0.8,
            "conflict_flag": false,
            "supporting_sources": ["source-1", "source-2"]
        }}
    ]
}}

ANALYZE:"""

    QUALITY_CHECK_TEMPLATE = """Review the following content for quality and accuracy issues.

Check for:
1. Unsupported claims (statements without source backing)
2. Contradictions between different parts
3. Vague or overly broad statements
4. Missing citations for specific facts

CONTENT:
{content}

SOURCES USED:
{sources}

Identify any quality issues and suggest improvements:"""


class ContentStructurer:
    """Structures content into sections and formats."""
    
    def __init__(self):
        """Initialize content structurer."""
        pass
    
    def structure_content(
        self, 
        content: str, 
        format: OutputFormat
    ) -> List[ContentSection]:
        """Structure content into sections based on format."""
        try:
            if format == OutputFormat.DETAILED_REPORT:
                return self._structure_detailed_report(content)
            elif format == OutputFormat.EXECUTIVE_SUMMARY:
                return self._structure_executive_summary(content)
            elif format == OutputFormat.FAQ:
                return self._structure_faq(content)
            elif format == OutputFormat.COMPARATIVE_ANALYSIS:
                return self._structure_comparative_analysis(content)
            elif format == OutputFormat.BULLET_POINTS:
                return self._structure_bullet_points(content)
            else:
                return self._structure_generic(content)
                
        except Exception as e:
            logger.warning(f"Error structuring content: {e}")
            return self._structure_generic(content)
    
    def _structure_detailed_report(self, content: str) -> List[ContentSection]:
        """Structure content as detailed report with sections."""
        sections = []
        
        # Split by markdown headers
        parts = content.split('\n## ')
        
        for i, part in enumerate(parts):
            if i == 0:
                # First part might not have ##
                if part.startswith('# '):
                    title = part.split('\n')[0].replace('# ', '')
                    section_content = '\n'.join(part.split('\n')[1:])
                else:
                    title = "Introduction"
                    section_content = part
            else:
                lines = part.split('\n')
                title = lines[0]
                section_content = '\n'.join(lines[1:])
            
            if title and section_content.strip():
                sections.append(ContentSection(
                    id=f"section-{i+1}",
                    title=title,
                    content=section_content.strip(),
                    start_pos=0,  # Would need more sophisticated calculation
                    end_pos=len(section_content),
                    subsections=[],
                    assertion_ids=[]
                ))
        
        return sections
    
    def _structure_executive_summary(self, content: str) -> List[ContentSection]:
        """Structure executive summary with key points."""
        sections = []
        
        # Look for common executive summary sections
        section_markers = ['## Executive Summary', '## Key Findings', '## Recommendations', '## Conclusion']
        
        current_section = {"title": "Executive Summary", "content": ""}
        
        lines = content.split('\n')
        for line in lines:
            if line.startswith('## '):
                # Save previous section
                if current_section["content"].strip():
                    sections.append(ContentSection(
                        id=f"section-{len(sections)+1}",
                        title=current_section["title"],
                        content=current_section["content"].strip(),
                        start_pos=0,
                        end_pos=len(current_section["content"]),
                        subsections=[],
                        assertion_ids=[]
                    ))
                
                # Start new section
                current_section = {
                    "title": line.replace('## ', ''),
                    "content": ""
                }
            else:
                current_section["content"] += line + '\n'
        
        # Add final section
        if current_section["content"].strip():
            sections.append(ContentSection(
                id=f"section-{len(sections)+1}",
                title=current_section["title"],
                content=current_section["content"].strip(),
                start_pos=0,
                end_pos=len(current_section["content"]),
                subsections=[],
                assertion_ids=[]
            ))
        
        return sections
    
    def _structure_faq(self, content: str) -> List[ContentSection]:
        """Structure FAQ content."""
        sections = []
        
        # Split by Q: or Question:
        qa_pairs = []
        current_qa = None
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('**Q:') or 'Question' in line:
                if current_qa:
                    qa_pairs.append(current_qa)
                current_qa = {"question": line, "answer": ""}
            elif line.startswith('A:') or line.startswith('**A:') or current_qa:
                if current_qa:
                    if line.startswith('A:') or line.startswith('**A:'):
                        current_qa["answer"] = line
                    else:
                        current_qa["answer"] += ' ' + line
        
        if current_qa:
            qa_pairs.append(current_qa)
        
        # Convert to sections
        for i, qa in enumerate(qa_pairs):
            sections.append(ContentSection(
                id=f"qa-{i+1}",
                title=qa["question"],
                content=qa["answer"],
                start_pos=0,
                end_pos=len(qa["answer"]),
                subsections=[],
                assertion_ids=[]
            ))
        
        return sections
    
    def _structure_comparative_analysis(self, content: str) -> List[ContentSection]:
        """Structure comparative analysis."""
        # Similar to detailed report but look for comparison markers
        return self._structure_detailed_report(content)
    
    def _structure_bullet_points(self, content: str) -> List[ContentSection]:
        """Structure bullet point content."""
        sections = []
        
        # Group bullet points by headers
        current_section = {"title": "Key Points", "content": ""}
        
        lines = content.split('\n')
        for line in lines:
            if line.startswith('## ') or line.startswith('# '):
                # Save previous section
                if current_section["content"].strip():
                    sections.append(ContentSection(
                        id=f"section-{len(sections)+1}",
                        title=current_section["title"],
                        content=current_section["content"].strip(),
                        start_pos=0,
                        end_pos=len(current_section["content"]),
                        subsections=[],
                        assertion_ids=[]
                    ))
                
                # Start new section
                current_section = {
                    "title": line.replace('## ', '').replace('# ', ''),
                    "content": ""
                }
            else:
                current_section["content"] += line + '\n'
        
        # Add final section
        if current_section["content"].strip():
            sections.append(ContentSection(
                id=f"section-{len(sections)+1}",
                title=current_section["title"],
                content=current_section["content"].strip(),
                start_pos=0,
                end_pos=len(current_section["content"]),
                subsections=[],
                assertion_ids=[]
            ))
        
        return sections
    
    def _structure_generic(self, content: str) -> List[ContentSection]:
        """Generic content structuring."""
        return [ContentSection(
            id="section-1",
            title="Content",
            content=content,
            start_pos=0,
            end_pos=len(content),
            subsections=[],
            assertion_ids=[]
        )]


class AssertionExtractor:
    """Extracts and validates assertions from generated content."""
    
    def __init__(self, llm):
        """Initialize assertion extractor."""
        self.llm = llm
    
    async def extract_assertions(
        self, 
        content: str, 
        sources: List[Document],
        citation_manager: CitationManager
    ) -> List[Assertion]:
        """Extract factual assertions from content."""
        try:
            logger.debug("Extracting assertions from generated content")
            
            # Prepare sources context
            sources_context = ""
            for i, doc in enumerate(sources):
                sources_context += f"Source-{i+1}: {doc.page_content[:500]}...\n\n"
            
            # Create extraction prompt
            prompt = PromptTemplates.ASSERTION_EXTRACTION_TEMPLATE.format(
                content=content[:2000],  # Limit content length
                sources=sources_context[:1500]  # Limit sources context
            )
            
            # Get LLM response
            response = await self._get_llm_response(prompt)
            
            # Parse JSON response
            try:
                assertions_data = json.loads(response)
                assertions = []
                
                for i, assertion_data in enumerate(assertions_data.get('assertions', [])):
                    # Create supporting sources
                    supporting_sources = []
                    for source_id in assertion_data.get('supporting_sources', []):
                        source_idx = int(source_id.split('-')[1]) - 1
                        if 0 <= source_idx < len(sources):
                            supporting_sources.append(SupportingSource(
                                doc_id=sources[source_idx].metadata.get('document_id', f'doc-{source_idx}'),
                                chunk_id=sources[source_idx].metadata.get('chunk_id', f'chunk-{source_idx}'),
                                excerpt=sources[source_idx].page_content[:200],
                                relevance_score=0.8,  # Default relevance
                                position_in_chunk=0
                            ))
                    
                    assertion = Assertion(
                        id=assertion_data.get('id', f'assert-{i+1}'),
                        text=assertion_data.get('text', ''),
                        consensus_score=assertion_data.get('consensus_score', 0.5),
                        conflict_flag=assertion_data.get('conflict_flag', False),
                        supporting_sources=supporting_sources,
                        confidence_level=self._calculate_confidence_level(
                            assertion_data.get('consensus_score', 0.5),
                            len(supporting_sources)
                        ),
                        section_id=None
                    )
                    assertions.append(assertion)
                
                logger.debug(f"Extracted {len(assertions)} assertions")
                return assertions
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse assertion extraction response as JSON")
                return self._extract_assertions_heuristic(content, sources)
                
        except Exception as e:
            logger.warning(f"Error extracting assertions: {e}")
            return self._extract_assertions_heuristic(content, sources)
    
    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.warning(f"Error getting LLM response: {e}")
            return "{\"assertions\": []}"
    
    def _extract_assertions_heuristic(
        self, 
        content: str, 
        sources: List[Document]
    ) -> List[Assertion]:
        """Fallback heuristic assertion extraction."""
        assertions = []
        
        # Simple heuristic: find sentences with citation patterns
        import re
        sentences = re.split(r'[.!?]+', content)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Look for citation patterns
            if re.search(r'\[Source-\d+\]', sentence):
                assertions.append(Assertion(
                    id=f'assert-{i+1}',
                    text=sentence,
                    consensus_score=0.7,
                    conflict_flag=False,
                    supporting_sources=[],  # Would need more sophisticated extraction
                    confidence_level="Medium",
                    section_id=None
                ))
        
        return assertions[:10]  # Limit number of assertions
    
    def _calculate_confidence_level(self, consensus_score: float, num_sources: int) -> str:
        """Calculate confidence level based on consensus and sources."""
        if consensus_score >= 0.8 and num_sources >= 3:
            return "High"
        elif consensus_score >= 0.6 and num_sources >= 2:
            return "Medium"
        else:
            return "Low"


class SynthesisService:
    """
    Main synthesis service that orchestrates the entire pipeline.
    """
    
    def __init__(self, config: Optional[SynthesisConfig] = None):
        """Initialize synthesis service."""
        self.config = config or self._create_default_config()
        self.llm = self._initialize_llm()
        self.content_structurer = ContentStructurer()
        self.assertion_extractor = AssertionExtractor(self.llm)
        self.initialized = False
        
    def _create_default_config(self) -> SynthesisConfig:
        """Create default configuration from settings."""
        return SynthesisConfig(
            groq_api_key=settings.groq_api_key,
            groq_model_name=settings.groq_model_name,
            max_context_tokens=settings.max_context_tokens,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.response_timeout,
            enable_compression=settings.enable_contextual_compression,
            consensus_threshold=settings.synthesis_consensus_threshold,
            min_sources_per_assertion=settings.assertion_min_sources
        )
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize Groq LLM."""
        try:
            llm = ChatGroq(
                groq_api_key=self.config.groq_api_key,
                model_name=self.config.groq_model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            logger.info(f"Initialized Groq LLM: {self.config.groq_model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            raise
    
    async def initialize(self) -> None:
        """Initialize the synthesis service."""
        try:
            # Initialize IR system if not already done
            if not comprehensive_ir_system.initialized:
                await comprehensive_ir_system.initialize()
            
            self.initialized = True
            logger.info("Synthesis service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize synthesis service: {e}")
            raise
    
    async def generate(
        self,
        query: str,
        format: str = "executive_summary",
        style: str = "academic",
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        citation_style: str = "APA"
    ) -> Dict[str, Any]:
        """
        Generate synthesized content with full pipeline.
        
        Args:
            query: Search query
            format: Output format
            style: Output style
            k: Number of documents to retrieve
            filters: Additional filters
            citation_style: Citation style
            
        Returns:
            Complete synthesis response
        """
        if not self.initialized:
            await self.initialize()
        
        request = SynthesisRequest(
            query=query,
            format=OutputFormat(format),
            style=OutputStyle(style),
            k=k or self.config.max_context_tokens,
            filters=filters or {},
            citation_style=CitationStyle(citation_style)
        )
        
        return await self._process_synthesis_request(request)
    
    async def generate_markdown(self, **kwargs) -> Dict[str, Any]:
        """Generate markdown-focused synthesis."""
        result = await self.generate(**kwargs)
        return {
            "markdown": result.get("answer_markdown", ""),
            "metadata": result.get("metadata", {}),
            "sources": result.get("sources", [])
        }
    
    async def explain_assertion(self, assertion_id: str) -> Dict[str, Any]:
        """
        Explain a specific assertion with supporting evidence.
        
        This would require maintaining assertion state, which is beyond
        the current scope but could be implemented with a cache/database.
        """
        # Placeholder implementation
        return {
            "assertion_id": assertion_id,
            "explanation": "Assertion explanation not implemented",
            "supporting_sources": [],
            "confidence": "Unknown"
        }
    
    async def _process_synthesis_request(self, request: SynthesisRequest) -> Dict[str, Any]:
        """Process a complete synthesis request."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing synthesis request: {request.query}")
            
            if request.dry_run:
                return await self._process_dry_run(request)
            
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            documents = await self._retrieve_documents(request)
            retrieval_time = int((time.time() - retrieval_start) * 1000)
            
            if not documents:
                return self._create_empty_response(request, start_time)
            
            # Step 2: Generate content with LLM
            llm_start = time.time()
            generated_content = await self._generate_content(request, documents)
            llm_time = int((time.time() - llm_start) * 1000)
            
            # Step 3: Structure content
            sections = self.content_structurer.structure_content(
                generated_content, request.format
            )
            
            # Step 4: Extract assertions
            citation_manager = get_citation_manager(request.citation_style)
            assertions = await self.assertion_extractor.extract_assertions(
                generated_content, documents, citation_manager
            )
            
            # Step 5: Generate citations
            sources = self._process_sources(documents, citation_manager, request.citation_style)
            
            # Step 6: Quality check
            quality_check_result = await quality_checker.check_quality(
                generated_content, [s.dict() for s in sections], assertions, sources
            )
            
            # Step 7: Create final response
            processing_time = int((time.time() - start_time) * 1000)
            
            # Convert to HTML if needed
            html_content = None
            try:
                html_content = markdown.markdown(generated_content)
            except Exception:
                pass
            
            response = SynthesisResponse(
                query=request.query,
                answer_markdown=generated_content,
                answer_text=self._markdown_to_text(generated_content),
                answer_html=html_content,
                format=request.format,
                style=request.style,
                sections=sections,
                assertions=assertions,
                sources=sources,
                highlights=self._extract_highlights(generated_content, documents),
                warnings=self._generate_warnings(quality_check_result, assertions, sources),
                metadata={
                    "retrieval_time_ms": retrieval_time,
                    "llm_time_ms": llm_time,
                    "quality_check": quality_check_result.dict(),
                    "citation_stats": citation_manager.get_citation_stats()
                },
                processing_time_ms=processing_time,
                tokens_used=self._estimate_tokens_used(generated_content)
            )
            
            logger.info(f"Synthesis completed in {processing_time}ms")
            return response.dict()
            
        except Exception as e:
            logger.error(f"Error in synthesis processing: {e}")
            return self._create_error_response(request, str(e), start_time)
    
    async def _retrieve_documents(self, request: SynthesisRequest) -> List[Document]:
        """Retrieve relevant documents using the IR system."""
        try:
            # Use comprehensive IR system for retrieval
            from app.models.schemas import IRSearchRequest, RetrievalMode
            
            ir_request = IRSearchRequest(
                query=request.query,
                retrieval_mode=RetrievalMode.HYBRID,
                max_results=request.k or self.config.max_context_tokens // 500,  # Estimate docs per context
                similarity_threshold=0.7,
                document_ids=None,
                filters=request.filters
            )
            
            ir_response = await comprehensive_ir_system.search(ir_request)
            
            # Convert IR results to LangChain documents
            documents = []
            for result in ir_response.results:
                doc = Document(
                    page_content=result.content_snippet,
                    metadata={
                        "document_id": result.document_id,
                        "chunk_id": result.chunk_id,
                        "similarity_score": result.final_score,
                        "source": result.source_metadata.get("source", ""),
                        "title": result.source_metadata.get("title", ""),
                        "author": result.source_metadata.get("author", ""),
                        "created_at": result.source_metadata.get("created_at", ""),
                        "url": result.source_metadata.get("url", "")
                    }
                )
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for synthesis")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def _generate_content(
        self, 
        request: SynthesisRequest, 
        documents: List[Document]
    ) -> str:
        """Generate content using LLM with retrieved documents."""
        try:
            # Prepare context from documents
            context = ""
            for i, doc in enumerate(documents):
                context += f"Source-{i+1}:\n{doc.page_content}\n\n"
            
            # Create synthesis prompt
            prompt = PromptTemplates.SYNTHESIS_TEMPLATE.format(
                query=request.query,
                format=request.format.value.replace('_', ' ').title(),
                style=request.style.value.title(),
                context=context[:self.config.max_context_tokens - 1000]  # Leave room for prompt
            )
            
            # Generate content
            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            content = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"Generated {len(content)} characters of content")
            return content
            
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return f"Error generating content: {e}\n\nNot enough information available to provide a comprehensive response."
    
    def _process_sources(
        self, 
        documents: List[Document], 
        citation_manager: CitationManager,
        citation_style: CitationStyle
    ) -> List[Dict[str, Any]]:
        """Process documents into citation format."""
        citations = []
        
        for doc in documents:
            metadata = SourceMetadata(
                doc_id=doc.metadata.get("document_id", "unknown"),
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                title=doc.metadata.get("title"),
                author=doc.metadata.get("author"),
                source=doc.metadata.get("source"),
                date=doc.metadata.get("created_at"),
                credibility_score=doc.metadata.get("similarity_score"),
                url=doc.metadata.get("url")
            )
            
            citation = citation_manager.add_source(metadata)
            citations.append(citation.dict())
        
        return citations
    
    def _extract_highlights(
        self, 
        content: str, 
        documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Extract highlighted passages from content."""
        highlights = []
        
        # Simple implementation: find quoted text that matches sources
        import re
        quoted_texts = re.findall(r'"([^"]+)"', content)
        
        for i, quote in enumerate(quoted_texts[:5]):  # Limit highlights
            # Try to match with source documents
            best_match = None
            best_score = 0
            
            for doc in documents:
                if quote.lower() in doc.page_content.lower():
                    best_match = doc
                    break
            
            if best_match:
                highlights.append({
                    "chunk_id": best_match.metadata.get("chunk_id", f"chunk-{i}"),
                    "snippet": quote,
                    "position": {"start": 0, "end": len(quote)},
                    "highlight_type": "quote",
                    "relevance_score": best_match.metadata.get("similarity_score", 0.5)
                })
        
        return highlights
    
    def _generate_warnings(
        self, 
        quality_check: QualityCheck, 
        assertions: List[Assertion],
        sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate warnings based on quality check and content analysis."""
        warnings = []
        
        # Quality-based warnings
        if quality_check.overall_quality == "Low":
            warnings.append("Content quality is low - review for accuracy")
        
        if quality_check.citation_coverage < 0.5:
            warnings.append("Low citation coverage - many claims lack supporting sources")
        
        if quality_check.hallucination_flags:
            warnings.extend(quality_check.hallucination_flags)
        
        # Assertion-based warnings
        low_confidence_assertions = sum(1 for a in assertions if a.confidence_level == "Low")
        if low_confidence_assertions > len(assertions) * 0.3:
            warnings.append(f"{low_confidence_assertions} assertions have low confidence")
        
        conflicted_assertions = sum(1 for a in assertions if a.conflict_flag)
        if conflicted_assertions > 0:
            warnings.append(f"{conflicted_assertions} assertions have conflicting information")
        
        # Source-based warnings
        if len(sources) < 3:
            warnings.append("Limited number of sources - consider broadening search")
        
        return warnings
    
    def _markdown_to_text(self, markdown_content: str) -> str:
        """Convert markdown to plain text."""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'#+\s+', '', markdown_content)  # Headers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines
        
        return text.strip()
    
    def _estimate_tokens_used(self, content: str) -> int:
        """Estimate tokens used (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters for English
        return len(content) // 4
    
    async def _process_dry_run(self, request: SynthesisRequest) -> Dict[str, Any]:
        """Process dry run request."""
        documents = await self._retrieve_documents(request)
        
        chunk_info = []
        for doc in documents:
            chunk_info.append({
                "chunk_id": doc.metadata.get("chunk_id"),
                "document_id": doc.metadata.get("document_id"),
                "content_preview": doc.page_content[:200] + "...",
                "similarity_score": doc.metadata.get("similarity_score"),
                "metadata": doc.metadata
            })
        
        dry_run_result = DryRunResult(
            query=request.query,
            retrieved_chunks=chunk_info,
            chunk_selection_reasoning=[
                f"Selected {len(documents)} chunks based on semantic similarity and relevance",
                f"Filtered by similarity threshold and document quality",
                f"Limited to top {request.k or 10} results"
            ],
            estimated_context_size=sum(len(doc.page_content) for doc in documents),
            compression_applied=request.k and len(documents) > request.k,
            warnings=[]
        )
        
        return dry_run_result.dict()
    
    def _create_empty_response(
        self, 
        request: SynthesisRequest, 
        start_time: float
    ) -> Dict[str, Any]:
        """Create response for empty results."""
        processing_time = int((time.time() - start_time) * 1000)
        
        return SynthesisResponse(
            query=request.query,
            answer_markdown="# No Information Available\n\nNot enough information available to answer this query.",
            answer_text="Not enough information available to answer this query.",
            answer_html="<h1>No Information Available</h1><p>Not enough information available to answer this query.</p>",
            format=request.format,
            style=request.style,
            sections=[],
            assertions=[],
            sources=[],
            highlights=[],
            warnings=["No relevant documents found for the query"],
            metadata={},
            processing_time_ms=processing_time,
            tokens_used=0
        ).dict()
    
    def _create_error_response(
        self, 
        request: SynthesisRequest, 
        error_message: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Create response for error cases."""
        processing_time = int((time.time() - start_time) * 1000)
        
        return SynthesisResponse(
            query=request.query,
            answer_markdown=f"# Error\n\nAn error occurred during synthesis: {error_message}",
            answer_text=f"An error occurred during synthesis: {error_message}",
            answer_html=f"<h1>Error</h1><p>An error occurred during synthesis: {error_message}</p>",
            format=request.format,
            style=request.style,
            sections=[],
            assertions=[],
            sources=[],
            highlights=[],
            warnings=[f"Synthesis error: {error_message}"],
            metadata={"error": error_message},
            processing_time_ms=processing_time,
            tokens_used=0
        ).dict()


# Global synthesis service instance
synthesis_service = SynthesisService()
