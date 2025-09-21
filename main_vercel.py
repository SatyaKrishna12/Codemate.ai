"""
Simplified Deep Researcher Agent for Vercel Deployment
Optimized for serverless functions with reduced dependencies
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import tempfile
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deep Researcher Agent",
    description="AI-powered document analysis and research assistant",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Simple in-memory session storage (replace with database in production)
sessions = {}

class ChatResponse:
    def __init__(self, content: str, response_type: str = "text"):
        self.content = content
        self.type = response_type
        self.timestamp = datetime.utcnow().isoformat()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page"""
    try:
        if os.path.exists("static/index.html"):
            with open("static/index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
        else:
            return HTMLResponse("""
            <html>
                <head><title>Deep Researcher Agent</title></head>
                <body>
                    <h1>🧠 Deep Researcher Agent</h1>
                    <p>AI-powered document analysis and research assistant</p>
                    <p><a href="/chat">Go to Chat Interface</a></p>
                    <p><a href="/api/v1/health">Health Check</a></p>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving root page: {e}")
        return HTMLResponse("<h1>Deep Researcher Agent</h1><p>Service is running</p>")

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface():
    """Serve the chat interface"""
    try:
        if os.path.exists("static/chat_interface.html"):
            with open("static/chat_interface.html", "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())
        else:
            return HTMLResponse("""
            <html>
                <head><title>Chat Interface</title></head>
                <body>
                    <h1>Chat Interface</h1>
                    <p>Chat interface will be available soon!</p>
                </body>
            </html>
            """)
    except Exception as e:
        logger.error(f"Error serving chat interface: {e}")
        return HTMLResponse("<h1>Chat Interface</h1><p>Coming soon!</p>")

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Deep Researcher Agent",
        "version": "1.0.0",
        "platform": "Vercel",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/researcher/start")
async def start_session():
    """Start a new research session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "documents": [],
        "messages": []
    }
    
    return {
        "session_id": session_id,
        "status": "active",
        "message": "Research session started successfully"
    }

@app.post("/api/v1/researcher/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    message: Optional[str] = Form("Please analyze the uploaded documents.")
):
    """Upload and process documents (simplified for serverless)"""
    try:
        if not session_id or session_id not in sessions:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        processed_files = []
        
        for file in files:
            # Simple file validation
            if file.size > 10 * 1024 * 1024:  # 10MB limit for serverless
                raise HTTPException(status_code=413, detail=f"File {file.filename} too large")
            
            # Save file info (in production, process content here)
            file_info = {
                "filename": file.filename,
                "size": file.size,
                "content_type": file.content_type,
                "uploaded_at": datetime.utcnow().isoformat()
            }
            
            sessions[session_id]["documents"].append(file_info)
            processed_files.append(file_info)
        
        # Simulate AI analysis response
        response_content = f"""📄 Successfully uploaded {len(files)} document(s):

{chr(10).join([f"• {f['filename']} ({f['size']} bytes)" for f in processed_files])}

🧠 **AI Analysis Summary:**
I've received your documents and they're ready for analysis. You can now ask me questions about:
- Document content and key insights
- Summaries and main themes
- Specific information extraction
- Cross-document comparisons

**What would you like to know about these documents?**"""
        
        return {
            "content": response_content,
            "type": "upload_summary",
            "files_processed": len(files),
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/researcher/chat-simple")
async def chat_simple(
    message: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    """Simple chat endpoint (simplified for serverless)"""
    try:
        if not session_id or session_id not in sessions:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Add user message to session
        sessions[session_id]["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Simple response generation (replace with actual AI integration)
        if "groq" in os.environ.get("GROQ_API_KEY", "").lower():
            # If Groq API key is available, could integrate here
            response_content = f"""🧠 **AI Research Assistant Response:**

Thank you for your question: "{message}"

I'm analyzing your query and will provide a comprehensive response. In this serverless deployment, I can help you with:

📚 **Document Analysis**: Extract insights from uploaded documents
🔍 **Research Assistance**: Answer questions and provide detailed explanations  
📊 **Data Synthesis**: Combine information from multiple sources
💡 **Recommendations**: Suggest next steps and related topics

For the full AI experience with document processing, please ensure your GROQ_API_KEY is properly configured.

**How else can I assist with your research?**"""
        else:
            response_content = f"""🧠 **Research Assistant Response:**

I received your message: "{message}"

**Current Status**: Simplified mode for Vercel deployment
**Documents in session**: {len(sessions[session_id]['documents'])}

To enable full AI capabilities, please configure the GROQ_API_KEY environment variable in your Vercel deployment settings.

**Available features in this mode:**
- Document upload and basic processing
- Session management
- Simple query responses

**How can I help you further?**"""
        
        # Add AI response to session
        sessions[session_id]["messages"].append({
            "role": "assistant", 
            "content": response_content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {"content": response_content, "type": "research_response"}
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/researcher/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "service": "Deep Researcher Agent"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "service": "Deep Researcher Agent"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)