from fastapi import FastAPI
from app.routers import compare

app = FastAPI(title="Visual Comparison Agent", version="1.0")

app.include_router(compare.router, prefix="/api")

@app.get("/")
async def root():
    return {"service": "Visual Comparison Agent", "status": "ok"}