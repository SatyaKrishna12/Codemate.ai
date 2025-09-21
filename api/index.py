"""
Vercel-compatible FastAPI application for Deep Researcher Agent
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

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

# Basic routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Deep Researcher Agent API",
        "status": "active",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "endpoints": [
            "/",
            "/health",
            "/api/health",
            "/test"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Deep Researcher Agent",
        "environment": os.getenv("VERCEL_ENV", "development")
    }

@app.get("/api/health")
async def api_health():
    """API health check"""
    return await health_check()

@app.get("/test")
async def test_endpoint():
    """Test endpoint"""
    return {
        "test": "success",
        "vercel": "working",
        "timestamp": datetime.now().isoformat(),
        "groq_configured": bool(os.getenv("GROQ_API_KEY"))
    }

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """Chat endpoint for research queries"""
    try:
        # Basic response for now
        response = {
            "response": f"Received your message: {message.message}",
            "chat_id": message.chat_id or "default",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/research")
async def research_endpoint(query: ResearchQuery):
    """Research endpoint"""
    try:
        response = {
            "query": query.query,
            "depth": query.depth,
            "result": f"Research query received: {query.query}",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Error handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Path {request.url.path} not found",
            "available_endpoints": ["/", "/health", "/test", "/api/chat", "/api/research"],
            "timestamp": datetime.now().isoformat()
        }
    )

# Vercel handler - This is important for Vercel deployment
handler = app