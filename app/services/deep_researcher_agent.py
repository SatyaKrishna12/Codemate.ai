"""
Deep Researcher Agent - Unified conversational research assistant.
Handles document processing, querying, and synthesis in a single interface.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from fastapi import UploadFile
from groq import Groq

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import ResearcherAgentException
from app.models.schemas import Document, DocumentChunk, ProcessingStatus
from app.services.document_processor import DocumentProcessor
from app.services.simple_synthesis_service import SimpleSynthesisService
from app.services.query_analyzer import QueryAnalyzer
from app.services.query_decomposer import QueryDecomposer
from app.services.context_manager import ContextManager
# from app.services.vector_store import VectorStore  # Temporarily disabled due to dependency issues

logger = get_logger(__name__)


class ConversationContext:
    """Tracks conversation state and history."""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.messages: List[Dict[str, Any]] = []
        self.uploaded_documents: Dict[str, Document] = {}
        self.processed_chunks: Dict[str, List[DocumentChunk]] = {}
        self.vector_store = None  # Will be initialized when needed
        self.created_at = datetime.utcnow()
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        })
        
    def get_recent_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context as a string."""
        recent_messages = self.messages[-max_messages:]
        context_parts = []
        
        for msg in recent_messages:
            context_parts.append(f"{msg['role'].title()}: {msg['content']}")
            
        return "\n".join(context_parts)


class DeepResearcherAgent:
    """
    Unified research assistant that handles the complete workflow:
    - Document upload and processing
    - Query analysis and decomposition  
    - Semantic search and retrieval
    - LLM-powered synthesis and response generation
    """
    
    def __init__(self):
        """Initialize the Deep Researcher Agent."""
        self.document_processor = DocumentProcessor()
        self.synthesis_service = SimpleSynthesisService()
        self.query_analyzer = QueryAnalyzer()
        self.query_decomposer = QueryDecomposer()
        self.context_manager = ContextManager()
        # self.vector_store = VectorStore()  # Temporarily disabled due to dependency issues
        
        # Active conversations
        self.conversations: Dict[str, ConversationContext] = {}
        
        logger.info("Deep Researcher Agent initialized")
        
    async def start_conversation(self) -> str:
        """Start a new conversation session."""
        context = ConversationContext()
        self.conversations[context.session_id] = context
        
        welcome_message = """🔬 **Welcome to Deep Researcher Agent!** 

I'm your unified research assistant. I can help you:

📄 **Upload & Process Documents** - I'll automatically extract text, create semantic chunks, and build searchable indexes
🔍 **Answer Questions** - Ask me anything about your documents or general topics  
📊 **Generate Reports** - Create summaries, comparisons, and detailed analyses
💬 **Continuous Dialogue** - Follow-up questions automatically use previous context

**Ready to get started?** 
- Upload documents by sharing files
- Ask questions about uploaded content
- Request specific analysis formats (summary, detailed report, comparison, etc.)

What would you like to explore today?"""

        context.add_message("assistant", welcome_message)
        
        logger.info(f"New conversation started: {context.session_id}")
        return context.session_id
        
    async def process_message(
        self, 
        session_id: str, 
        message: str, 
        files: Optional[List[UploadFile]] = None
    ) -> Dict[str, Any]:
        """
        Process a user message, handling document uploads and queries.
        
        Args:
            session_id: Conversation session ID
            message: User message text
            files: Optional uploaded files
            
        Returns:
            Response with content, sources, and metadata
        """
        try:
            # Get conversation context
            context = self.conversations.get(session_id)
            if not context:
                raise ResearcherAgentException(f"Invalid session ID: {session_id}")
                
            # Add user message to context
            context.add_message("user", message)
            
            # Handle file uploads first
            upload_results = []
            if files:
                upload_results = await self._handle_file_uploads(context, files)
                
            # Determine intent and generate response
            response = await self._generate_response(context, message, upload_results)
            
            # Add assistant response to context
            context.add_message("assistant", response["content"], response.get("metadata"))
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_response = {
                "content": f"I encountered an error: {str(e)}. Please try again.",
                "type": "error",
                "sources": [],
                "metadata": {"error": str(e)}
            }
            return error_response
            
    async def _handle_file_uploads(
        self, 
        context: ConversationContext, 
        files: List[UploadFile]
    ) -> List[Dict[str, Any]]:
        """Handle document uploads and processing."""
        upload_results = []
        
        for file in files:
            try:
                # Save uploaded file
                file_id = str(uuid.uuid4())
                upload_dir = Path(settings.uploads_dir)
                upload_dir.mkdir(exist_ok=True)
                
                file_path = upload_dir / f"{file_id}_{file.filename}"
                
                # Save file content
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                    
                # Process document
                document = await self.document_processor.process_document(
                    str(file_path), 
                    file.filename, 
                    len(content)
                )
                
                # Extract text for chunking from the metadata returned by process_document
                # The process_document method now returns the Document object with metadata
                # We need to extract text again for chunking
                text_content, _ = await self.document_processor._extract_text_and_metadata(
                    str(file_path), 
                    file.filename, 
                    len(content),
                    document.metadata.file_type
                )
                
                # Create chunks
                chunks = await self.document_processor.create_chunks(document, text_content)
                
                # TODO: Add vector store integration once dependencies are resolved
                # Initialize vector store if needed
                # if not hasattr(self.vector_store, '_index') or self.vector_store._index is None:
                #     await self.vector_store.initialize()
                
                # Store chunks in vector store for semantic search
                # chunk_texts = [chunk.content for chunk in chunks]
                # chunk_metadata = [
                #     {
                #         "chunk_id": chunk.chunk_id,
                #         "document_id": document.id,
                #         "filename": file.filename,
                #         "chunk_index": chunk.chunk_index
                #     }
                #     for chunk in chunks
                # ]
                
                # Add chunks to vector store
                # await self.vector_store.add_texts(chunk_texts, metadata=chunk_metadata)
                
                # Store in context
                context.uploaded_documents[document.id] = document
                context.processed_chunks[document.id] = chunks
                
                # Create success result
                result = {
                    "status": "success",
                    "document_id": document.id,
                    "filename": file.filename,
                    "size": len(content),
                    "chunks_created": len(chunks),
                    "pages": document.metadata.page_count or "N/A",
                    "summary": f"Successfully processed {file.filename} ({len(content):,} bytes, {len(chunks)} chunks) and ready for search"
                }
                
                upload_results.append(result)
                
                logger.info(f"Document processed successfully: {file.filename} -> {document.id}")
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                upload_results.append({
                    "status": "error",
                    "filename": file.filename,
                    "error": str(e)
                })
                
        return upload_results
        
    async def _generate_response(
        self, 
        context: ConversationContext, 
        message: str, 
        upload_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate an appropriate response based on context and intent."""
        
        # If files were uploaded, provide upload summary
        if upload_results:
            return await self._create_upload_response(upload_results)
            
        # Check if user is asking a greeting or wants to start a conversation
        greeting_keywords = ["hello", "hi", "hey", "start", "begin", "help"]
        is_greeting = any(keyword in message.lower() for keyword in greeting_keywords) and len(message.split()) < 5
        
        # If it's just a greeting and no documents are uploaded, suggest uploading
        if is_greeting and not context.uploaded_documents:
            return {
                "content": """I'd be happy to help you research! However, I don't see any documents uploaded yet. 

**To get the most value from our conversation:**
1. 📄 Upload your documents (PDF, DOCX, TXT, or Markdown files)
2. 🔍 Ask specific questions about the content
3. 📊 Request analysis in your preferred format

**Or feel free to ask me general questions** - I can help with research topics even without uploaded documents!

What would you like to explore?""",
                "type": "suggestion",
                "sources": [],
                "metadata": {"intent": "document_upload_suggestion"}
            }
            
        # Always analyze query and generate research response for substantive questions
        return await self._create_research_response(context, message)
        
    async def _create_upload_response(self, upload_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create response for file upload results."""
        successful_uploads = [r for r in upload_results if r["status"] == "success"]
        failed_uploads = [r for r in upload_results if r["status"] == "error"]
        
        response_parts = ["📄 **Document Processing Complete!**\n"]
        
        if successful_uploads:
            response_parts.append("✅ **Successfully Processed:**")
            for result in successful_uploads:
                response_parts.append(
                    f"• **{result['filename']}**: {result['chunks_created']} chunks created "
                    f"({result['size']:,} bytes, {result['pages']} pages)"
                )
                
        if failed_uploads:
            response_parts.append("\n❌ **Processing Errors:**")
            for result in failed_uploads:
                response_parts.append(f"• **{result['filename']}**: {result['error']}")
                
        response_parts.extend([
            "\n🔍 **Ready for Questions!**",
            "Your documents are now indexed and searchable. Ask me:",
            "• Specific questions about the content",
            "• For summaries or key insights", 
            "• To compare information across documents",
            "• For detailed analysis on particular topics",
            "",
            "What would you like to explore in these documents?"
        ])
        
        return {
            "content": "\n".join(response_parts),
            "type": "upload_summary",
            "sources": [{"filename": r["filename"], "chunks": r["chunks_created"]} for r in successful_uploads],
            "metadata": {
                "successful_uploads": len(successful_uploads),
                "failed_uploads": len(failed_uploads)
            }
        }
        
    async def _create_research_response(
        self, 
        context: ConversationContext, 
        message: str
    ) -> Dict[str, Any]:
        """Create a research response using query analysis and synthesis."""
        
        try:
            # Analyze the query
            query_analysis = await self.query_analyzer.analyze_query(message)
            
            # Decompose complex queries
            sub_queries = await self.query_decomposer.decompose_query(query_analysis)
            
            # Gather relevant chunks from uploaded documents (if any)
            relevant_chunks = await self._retrieve_relevant_chunks(context, message, sub_queries)
            
            # Build context for synthesis
            conversation_context = context.get_recent_context(5)
            
            # Prepare sources list for synthesis
            sources_list = [
                {
                    "title": context.uploaded_documents[chunk.document_id].filename,
                    "content": chunk.content
                }
                for chunk in relevant_chunks
            ] if relevant_chunks else []
            
            # Generate synthesis using Groq LLM (quick synthesis for better compatibility)
            synthesis_response = await self.synthesis_service.quick_synthesis(
                message, sources_list
            )
            
            # Format final response
            response = {
                "content": synthesis_response,
                "type": "research_response",
                "sources": self._format_sources(relevant_chunks, context) if relevant_chunks else [],
                "metadata": {
                    "query_type": query_analysis.query_type.value if query_analysis.query_type else "general",
                    "chunks_used": len(relevant_chunks),
                    "sub_queries": len(sub_queries),
                    "confidence": 0.8,
                    "has_documents": len(context.uploaded_documents) > 0
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating research response: {str(e)}")
            return {
                "content": f"I encountered an issue while researching your question: {str(e)}. Could you please rephrase or try a different approach?",
                "type": "error",
                "sources": [],
                "metadata": {"error": str(e)}
            }
            
    async def _retrieve_relevant_chunks(
        self, 
        context: ConversationContext, 
        query: str, 
        sub_queries: List[Any]
    ) -> List[DocumentChunk]:
        """Retrieve relevant document chunks using text matching (vector search temporarily disabled)."""
        
        relevant_chunks = []
        
        # TODO: Add vector search once dependencies are resolved
        # try:
        #     # Try semantic search using vector store if available and initialized
        #     if (hasattr(self.vector_store, '_index') and 
        #         self.vector_store._index is not None and 
        #         self.vector_store._index.ntotal > 0):
        #         
        #         logger.info(f"Using vector search for query: {query}")
        #         
        #         # Perform semantic search
        #         search_results = await self.vector_store.search(query, limit=10)
        #         
        #         # Convert search results to DocumentChunk objects with relevance scores
        #         for result in search_results:
        #             # Find the corresponding chunk in context
        #             for document_id, chunks in context.processed_chunks.items():
        #                 for chunk in chunks:
        #                     if chunk.chunk_id == result.metadata.get("chunk_id"):
        #                         chunk.relevance_score = result.score
        #                         relevant_chunks.append(chunk)
        #                         break
        #                         
        #         if relevant_chunks:
        #             logger.info(f"Vector search found {len(relevant_chunks)} relevant chunks")
        #             return relevant_chunks
        #             
        # except Exception as e:
        #     logger.warning(f"Vector search failed, falling back to text matching: {str(e)}")
        
        # Use text matching for chunk retrieval
        logger.info("Using text matching for chunk retrieval")
        
        query_terms = query.lower().split()
        
        # Add sub-query terms
        for sub_query in sub_queries:
            query_terms.extend(sub_query.text.lower().split())
            
        # Remove duplicates and common words
        query_terms = list(set(query_terms))
        common_words = {"the", "and", "or", "but", "is", "are", "was", "were", "a", "an", "to", "for", "of", "in", "on", "at"}
        query_terms = [term for term in query_terms if term not in common_words and len(term) > 2]
        
        # Search through all chunks
        for document_id, chunks in context.processed_chunks.items():
            for chunk in chunks:
                # Calculate relevance score
                chunk_text = chunk.content.lower()
                matches = sum(1 for term in query_terms if term in chunk_text)
                
                if matches > 0:
                    relevance_score = matches / len(query_terms)
                    relevant_chunks.append((chunk, relevance_score))
                    
        # Sort by relevance and return top chunks
        relevant_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in relevant_chunks[:10]]  # Return top 10 most relevant chunks
        
    def _format_sources(
        self, 
        chunks: List[DocumentChunk], 
        context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """Format source information for response."""
        sources = []
        
        for chunk in chunks:
            document = context.uploaded_documents.get(chunk.document_id)
            if document:
                sources.append({
                    "filename": document.filename,
                    "chunk_id": chunk.chunk_id,
                    "preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                })
                
        return sources
        
    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        context = self.conversations.get(session_id)
        if not context:
            return []
            
        return context.messages
        
    async def list_uploaded_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """List all documents uploaded in a session."""
        context = self.conversations.get(session_id)
        if not context:
            return []
            
        documents = []
        for document in context.uploaded_documents.values():
            documents.append({
                "id": document.id,
                "filename": document.filename,
                "upload_time": document.upload_timestamp.isoformat(),
                "status": document.processing_status.value,
                "chunks": len(context.processed_chunks.get(document.id, [])),
                "metadata": {
                    "file_size": document.metadata.file_size,
                    "page_count": document.metadata.page_count,
                    "word_count": document.metadata.word_count
                }
            })
            
        return documents


# Global agent instance
deep_researcher_agent = DeepResearcherAgent()
