"""
Integration guide for adding the Comprehensive IR System to your existing FastAPI application.

This file shows how to integrate the new IR routes with your existing app structure.
"""

# To integrate the IR system with your existing FastAPI app, add these lines to your main.py:

"""
# Add this import at the top of your main.py or wherever you define your FastAPI app
from app.api.ir_routes import router as ir_router

# Add this line after creating your FastAPI app instance
app.include_router(ir_router)

# Example:
from fastapi import FastAPI
from app.api.ir_routes import router as ir_router

app = FastAPI(title="Researcher Agent with IR System")

# Include your existing routes
# app.include_router(your_existing_routes)

# Include the new IR routes
app.include_router(ir_router)

# Your existing startup events and other configurations...
"""

# Alternative: If you have a separate API router setup, you can add to your api/__init__.py:

"""
# In app/api/__init__.py (if it exists):
from .ir_routes import router as ir_router

# Then include it in your main router
"""

# Testing the IR System:
# Once integrated, you can test the IR system with these endpoints:

test_endpoints = {
    "comprehensive_search_post": "POST /ir/search",
    "comprehensive_search_get": "GET /ir/search?query=your_query&retrieval_mode=hybrid",
    "system_status": "GET /ir/status", 
    "initialize_system": "POST /ir/initialize",
    "capabilities": "GET /ir/capabilities"
}

# Example usage with curl:
example_curl_commands = [
    """
    # GET search (simple)
    curl "http://localhost:8003/ir/search?query=machine learning&retrieval_mode=hybrid&max_results=5"
    """,
    """
    # POST search (advanced)
    curl -X POST "http://localhost:8003/ir/search" \
         -H "Content-Type: application/json" \
         -d '{
           "query": "artificial intelligence applications",
           "retrieval_mode": "hybrid",
           "max_results": 10,
           "similarity_threshold": 0.7,
           "document_ids": null,
           "filters": {}
         }'
    """,
    """
    # Check system status
    curl "http://localhost:8003/ir/status"
    """,
    """
    # Get capabilities
    curl "http://localhost:8003/ir/capabilities"
    """
]

print("IR System Integration Guide")
print("=" * 50)
print("1. Add the IR router to your main FastAPI app")
print("2. Test the endpoints listed above")
print("3. The system will auto-initialize on first use")
print("4. Check /ir/status for system health")
print("5. Use /ir/capabilities to see all available features")
