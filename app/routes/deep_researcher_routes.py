"""
Unified conversational API routes for Deep Researcher Agent.
Provides seamless document upload, processing, and querying through a single chat interface.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.logging import get_logger
from app.services.deep_researcher_agent import deep_researcher_agent
from app.models.schemas import StandardResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/researcher", tags=["Deep Researcher Agent"])


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat messages."""
    session_id: Optional[str] = None
    message: str
    include_history: bool = True


class ChatResponse(BaseModel):
    """Response model for chat messages."""
    session_id: str
    content: str
    type: str  # "welcome", "upload_summary", "research_response", "error", "suggestion"
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    conversation_history: Optional[List[Dict[str, Any]]] = None


class SessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    message: str


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    session_id: str
    documents: List[Dict[str, Any]]


# API Endpoints
@router.post("/start", response_model=SessionResponse)
async def start_conversation() -> SessionResponse:
    """
    Start a new conversation session with the Deep Researcher Agent.
    
    Returns:
        SessionResponse with new session ID and welcome message
    """
    try:
        session_id = await deep_researcher_agent.start_conversation()
        
        return SessionResponse(
            session_id=session_id,
            message="New research session started successfully"
        )
        
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat_message(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[]),
    include_history: bool = Form(True)
) -> ChatResponse:
    """
    Send a message to the Deep Researcher Agent with optional file uploads.
    
    Args:
        message: User message text
        session_id: Optional existing session ID (creates new session if not provided)
        files: Optional uploaded files
        include_history: Whether to include conversation history in response
        
    Returns:
        ChatResponse with agent response and metadata
    """
    try:
        # Create new session if not provided
        if not session_id:
            session_id = await deep_researcher_agent.start_conversation()
            
        # Process message and files
        response = await deep_researcher_agent.process_message(
            session_id=session_id,
            message=message,
            files=files if files else None
        )
        
        # Get conversation history if requested
        conversation_history = None
        if include_history:
            conversation_history = await deep_researcher_agent.get_conversation_history(session_id)
            
        return ChatResponse(
            session_id=session_id,
            content=response["content"],
            type=response["type"],
            sources=response["sources"],
            metadata=response["metadata"],
            conversation_history=conversation_history
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.post("/upload", response_model=ChatResponse)
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
    message: Optional[str] = Form("Please process the uploaded documents.")
) -> ChatResponse:
    """
    Upload documents to an existing session.
    
    Args:
        session_id: Existing session ID
        files: List of files to upload
        message: Optional message to go with the upload
        
    Returns:
        ChatResponse with upload results
    """
    try:
        # Process uploaded files
        response = await deep_researcher_agent.process_message(
            session_id=session_id,
            message=message,
            files=files
        )
        
        return ChatResponse(
            session_id=session_id,
            content=response["content"],
            type=response["type"],
            sources=response["sources"],
            metadata=response["metadata"],
            conversation_history=None
        )
        
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload documents: {str(e)}")


@router.post("/chat-simple", response_model=Dict[str, Any])
async def chat_simple(request: ChatRequest) -> Dict[str, Any]:
    """
    Simple chat endpoint for text-only messages (no file upload).
    
    Args:
        request: ChatRequest with message and optional session ID
        
    Returns:
        Chat response with agent reply
    """
    try:
        # Create new session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = await deep_researcher_agent.start_conversation()
            
        # Process message
        response = await deep_researcher_agent.process_message(
            session_id=session_id,
            message=request.message,
            files=None
        )
        
        # Get conversation history if requested
        conversation_history = None
        if request.include_history:
            conversation_history = await deep_researcher_agent.get_conversation_history(session_id)
            
        return {
            "session_id": session_id,
            "content": response["content"],
            "type": response["type"],
            "sources": response["sources"],
            "metadata": response["metadata"],
            "conversation_history": conversation_history
        }
        
    except Exception as e:
        logger.error(f"Error processing simple chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.get("/session/{session_id}/history", response_model=List[Dict[str, Any]])
async def get_conversation_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Get conversation history for a specific session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of conversation messages with timestamps
    """
    try:
        history = await deep_researcher_agent.get_conversation_history(session_id)
        return history
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/session/{session_id}/documents", response_model=DocumentListResponse)
async def list_session_documents(session_id: str) -> DocumentListResponse:
    """
    List all documents uploaded in a specific session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        DocumentListResponse with uploaded documents info
    """
    try:
        documents = await deep_researcher_agent.list_uploaded_documents(session_id)
        
        return DocumentListResponse(
            session_id=session_id,
            documents=documents
        )
        
    except Exception as e:
        logger.error(f"Error listing session documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/health", response_model=StandardResponse)
async def health_check() -> StandardResponse:
    """Health check endpoint for the Deep Researcher Agent."""
    try:
        return StandardResponse(
            message="Deep Researcher Agent is healthy",
            data={
                "status": "healthy",
                "active_sessions": len(deep_researcher_agent.conversations),
                "components": {
                    "document_processor": "available",
                    "synthesis_service": "available", 
                    "query_analyzer": "available",
                    "query_decomposer": "available",
                    "context_manager": "available"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/sessions", response_model=List[Dict[str, Any]])
async def list_active_sessions() -> List[Dict[str, Any]]:
    """
    List all active conversation sessions.
    
    Returns:
        List of active sessions with metadata
    """
    try:
        sessions = []
        for session_id, context in deep_researcher_agent.conversations.items():
            sessions.append({
                "session_id": session_id,
                "created_at": context.created_at.isoformat(),
                "message_count": len(context.messages),
                "document_count": len(context.uploaded_documents),
                "last_activity": context.messages[-1]["timestamp"].isoformat() if context.messages else context.created_at.isoformat()
            })
            
        return sessions
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


# Demo endpoints for testing
@router.post("/demo/upload", response_model=Dict[str, Any])
async def demo_upload(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Demo endpoint to test document upload functionality.
    
    Args:
        files: List of files to upload
        
    Returns:
        Upload results
    """
    try:
        # Start new session
        session_id = await deep_researcher_agent.start_conversation()
        
        # Process files
        response = await deep_researcher_agent.process_message(
            session_id=session_id,
            message="Please process these uploaded documents.",
            files=files
        )
        
        return {
            "session_id": session_id,
            "response": response,
            "message": "Demo upload completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in demo upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo upload failed: {str(e)}")


@router.post("/demo/query", response_model=Dict[str, Any])
async def demo_query(
    query: str = Form(...),
    session_id: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Demo endpoint to test query functionality.
    
    Args:
        query: Query text
        session_id: Optional session ID (creates new if not provided)
        
    Returns:
        Query response
    """
    try:
        # Create session if needed
        if not session_id:
            session_id = await deep_researcher_agent.start_conversation()
            
        # Process query
        response = await deep_researcher_agent.process_message(
            session_id=session_id,
            message=query,
            files=None
        )
        
        return {
            "session_id": session_id,
            "query": query,
            "response": response,
            "message": "Demo query completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in demo query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo query failed: {str(e)}")
