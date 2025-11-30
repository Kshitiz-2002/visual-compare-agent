from fastapi import FastAPI

app = FastAPI(title="Visual Compare Agent", version="1.0")

@app.get("/")
async def root():
    return {"service":"Visual Compare Agent", "status":"ok"}