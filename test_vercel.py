from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from Vercel!", "status": "working"}

@app.get("/test")
async def test():
    return {"test": "success", "vercel": "working"}

# Vercel handler
handler = app