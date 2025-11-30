from fastapi import FastAPI
from app.routers import compare

app = FastAPI(
    title="Visual Comparison Agent",
    version="3.0",
    description="Moondream2-based visual diff agent (local)."
)

@app.get("/")
def status():
    return {
        "status": "running",
        "agent": True,
        "model": "vikhyatk/moondream2"
    }

app.include_router(compare.router, prefix="/api")
