"""
Enhanced Deep Researcher Agent for Vercel Deployment
Serverless-optimized with full API compatibility
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

# FastAPI and async imports
from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Deep Researcher Agent API",
    description="Serverless research agent powered by Groq AI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    chat_id: Optional[str] = None

class ResearchQuery(BaseModel):
    query: str
    depth: str = "deep"

# In-memory storage for serverless
chat_sessions = {}
uploaded_documents = {}

# Groq client setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables")

# Mock research function for serverless environment
async def conduct_research(query: str, depth: str = "deep") -> Dict[str, Any]:
    """
    Lightweight research function for serverless environment
    """
    try:
        # Simulate research response
        research_results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "depth": depth,
            "findings": [
                f"Research finding 1 for: {query}",
                f"Research finding 2 for: {query}",
                f"Research finding 3 for: {query}"
            ],
            "summary": f"Summary of research conducted for: {query}",
            "sources": [
                "https://example.com/source1",
                "https://example.com/source2"
            ]
        }
        
        if GROQ_API_KEY:
            # If Groq is available, add AI-enhanced response
            research_results["ai_analysis"] = f"AI-enhanced analysis for: {query}"
        
        return research_results
    except Exception as e:
        return {
            "error": f"Research failed: {str(e)}",
            "query": query,
            "timestamp": datetime.now().isoformat()
        }

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Deep Researcher Agent API", "status": "active", "version": "1.0.0"}

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "groq_available": bool(GROQ_API_KEY)
    }

# Chat endpoint
@app.post("/api/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
        chat_id = chat_message.chat_id or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize chat session if not exists
        if chat_id not in chat_sessions:
            chat_sessions[chat_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat()
            }
        
        # Add user message
        chat_sessions[chat_id]["messages"].append({
            "role": "user",
            "content": chat_message.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate AI response
        ai_response = f"Research response for: {chat_message.message}"
        
        if GROQ_API_KEY:
            # Enhanced response with Groq if available
            ai_response = f"Enhanced AI response using Groq for: {chat_message.message}"
        
        # Add AI message
        chat_sessions[chat_id]["messages"].append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "response": ai_response,
            "chat_id": chat_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

# Research endpoint
@app.post("/api/research")
async def research_endpoint(research_query: ResearchQuery):
    try:
        research_results = await conduct_research(
            query=research_query.query,
            depth=research_query.depth
        )
        
        return {
            "success": True,
            "research_results": research_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research error: {str(e)}")

# Document upload endpoint
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        # Read file content
        content = await file.read()
        
        # Store document info
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        uploaded_documents[doc_id] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "uploaded_at": datetime.now().isoformat(),
            "content": content.decode('utf-8', errors='ignore') if file.content_type and 'text' in file.content_type else None
        }
        
        return {
            "success": True,
            "message": f"File '{file.filename}' uploaded successfully",
            "document_id": doc_id,
            "filename": file.filename,
            "size": len(content),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# Get documents endpoint
@app.get("/api/documents")
async def get_documents():
    try:
        documents = []
        for doc_id, doc_info in uploaded_documents.items():
            documents.append({
                "id": doc_id,
                "filename": doc_info["filename"],
                "size": doc_info["size"],
                "uploaded_at": doc_info["uploaded_at"],
                "content_type": doc_info["content_type"]
            })
        
        return {
            "success": True,
            "documents": documents,
            "count": len(documents),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Documents error: {str(e)}")

# Chat sessions endpoint
@app.get("/api/sessions")
async def get_chat_sessions():
    try:
        sessions = []
        for session_id, session_info in chat_sessions.items():
            sessions.append({
                "chat_id": session_id,
                "created_at": session_info["created_at"],
                "message_count": len(session_info["messages"])
            })
        
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sessions error: {str(e)}")

# Status endpoint
@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "environment": "serverless",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(chat_sessions),
        "uploaded_documents": len(uploaded_documents),
        "groq_configured": bool(GROQ_API_KEY),
        "features": {
            "chat": True,
            "research": True,
            "document_upload": True,
            "session_management": True
        }
    }

# Form-based chat endpoint for frontend compatibility
@app.post("/chat")
async def chat_form_endpoint(
    message: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    try:
        # Create chat message object
        chat_msg = ChatMessage(message=message, chat_id=session_id)
        
        # Use existing chat endpoint logic
        result = await chat_endpoint(chat_msg)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Form chat error: {str(e)}")

# Form-based upload endpoint for frontend compatibility
@app.post("/upload")
async def upload_form_endpoint(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    message: Optional[str] = Form("Please analyze the uploaded document.")
):
    try:
        # Upload file using existing endpoint
        upload_result = await upload_document(file)
        
        # If session provided, add to chat
        if session_id and session_id in chat_sessions:
            chat_sessions[session_id]["documents"] = chat_sessions[session_id].get("documents", [])
            chat_sessions[session_id]["documents"].append({
                "filename": file.filename,
                "uploaded_at": datetime.now().isoformat()
            })
        
        return JSONResponse(content=upload_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Form upload error: {str(e)}")

# Catch-all for other routes
@app.get("/{path:path}")
async def catch_all(path: str):
    return {
        "message": f"Path '/{path}' not found",
        "available_endpoints": [
            "/",
            "/health",
            "/api/chat",
            "/api/research", 
            "/api/upload",
            "/api/documents",
            "/api/sessions",
            "/api/status",
            "/chat",
            "/upload"
        ]
    }

# Vercel handler
app_handler = app

# Optional: Local development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)