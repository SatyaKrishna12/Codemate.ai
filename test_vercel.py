from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from Vercel!", "status": "working"}

@app.get("/test")
async def test():
    return {"test": "success", "vercel": "working"}

@app.get("/api/test")
async def api_test():
    return {"api": "working", "endpoint": "test"}

# Vercel ASGI handler
def handler(scope, receive, send):
    return app(scope, receive, send)

# Alternative handler names for Vercel compatibility
app_handler = app
asgi_app = app