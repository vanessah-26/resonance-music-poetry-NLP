import json
from pathlib import Path
 
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from rag import poems_to_chroma_db, query_poems
from extract import extract_features_from_image
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles


SHEETS_FILE = "sheets/sheets_metadata.json"
SHEETS_DIR  = "sheets"

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    poems_to_chroma_db()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # edit my live Github page 
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory="sheets/audio"), name="audio")


def load_music_sheets() -> list[dict]:
     with open(SHEETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
     
# Request and Response models

class MatchRequest(BaseModel):
    sheet_id: str

class MusicFeatures(BaseModel):
    tempo_feel:      str
    key_feel:        str
    mood:            str
    tags:            str
 
class PoemResult(BaseModel):
    id:               str
    title:            str
    poet:             str
    language:         str
    original:         str
    translation:      str | None
    is_excerpt:       bool
    tags:             str
    similarity_score: float
 
class MatchResponse(BaseModel):
    sheet_title: str
    composer:    str
    audio_file:  str | None    
    features:    MusicFeatures
    poems:       list[PoemResult]


# API Endpoints
@app.get("/sheets")
def get_sheets():
    """
    Return the list of available sheets and used by the frontend to display the options. 
    """

    sheets = load_music_sheets()
    return [
        {
            "id":       s["id"],
            "title":    s["title"],
            "composer": s["composer"],
            "filename": s["filename"]
        }
        for s in sheets
    ]

@app.post("/match", response_model=MatchResponse)
def match_poems(request: MatchRequest):
    sheets = load_music_sheets()

    # Find the requested sheet 
    sheet = next((s for s in sheets if s["id"] == request.sheet_id), None)
    if not sheet:
        raise HTTPException(status_code=404, detail=f"Sheet '{request.sheet_id}' not found.")
 
    # Query RAG with pre-computed tags 
    poems = query_poems(sheet["tags"])
 
    return {
        "sheet_title": sheet["title"],
        "composer":    sheet["composer"],
         "audio_file":  sheet.get("audio_file"), 
        "features": {
            "tempo_feel": sheet.get("tempo_feel", ""),
            "key_feel":   sheet.get("key_feel", ""),
            "mood":       sheet.get("mood", ""),
            "tags":       sheet["tags"]
        },
        "poems": poems
    }

@app.post("/upload", response_model=MatchResponse)
async def upload_sheet(file: UploadFile = File(...)):
    """
    When user uploads a new music sheet:
    1. Call extract.py to get features and tags 
    2. Query ChromaDB RAG with extracted tags to find matching poems
    3. REturn extracted features and matched poems to display 
    """

    # Added
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    img_bytes = await file.read()

    # Extract features 
    try:
        features = extract_features_from_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting features: {e}")
    
    # Query RAG 
    poems = query_poems(features["tags"])

    return {
        "sheet_title": file.filename,
        "composer":    "Unknown",
        "audio_file":  None, 
        "features": {
            "tempo_feel": features.get("tempo_feel", ""),
            "key_feel":   features.get("key_feel", ""),
            "mood":       features.get("mood", ""),
            "tags":       features["tags"]
        },
        "poems": poems
    }

# health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}