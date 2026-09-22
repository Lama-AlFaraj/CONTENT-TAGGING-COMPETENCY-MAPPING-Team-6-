from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI(title="BeamData AI Hub", version="1.0")

V8_API_URL = os.getenv(
    "V8_API_URL",
    "http://beamdata-v8-api:8000/predict"
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "beamdata-ai-hub",
        "version": "1.0"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()

    response = requests.post(
        V8_API_URL,
        files={
            "file": (
                file.filename,
                content,
                file.content_type or "application/octet-stream"
            )
        },
        timeout=300
    )

    if response.status_code != 200:
        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "upstream": "beamdata-v8-api",
                "detail": response.text
            }
        )

    return {
        "ai_hub": "beamdata-ai-hub",
        "upstream": "beamdata-v8-api",
        "result": response.json()
    }
