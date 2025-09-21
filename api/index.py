from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing app
try:
    from main_simplified import app as main_app
    app = main_app
except ImportError:
    # Fallback minimal app if imports fail
    app = FastAPI(title="Deep Researcher Agent")
    
    @app.get("/")
    async def root():
        return {"message": "Deep Researcher Agent API - Vercel Deployment"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "platform": "vercel"}

# Configure CORS for Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel serverless function handler
def handler(request):
    return app