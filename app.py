import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import poems_to_chroma_db, query_poems

from dotenv import load_dotenv
load_dotenv()

SHEETS_FILE = "sheets/sheets_metadata.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once on server startup.
    Loads poems into ChromaDB — skips if already populated.
    """
    poems_to_chroma_db()
    yield


app = FastAPI(title="Resonance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # uupdate my live url
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_music_sheets() -> list[dict]:
    with open(SHEETS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# Request and Response models 

class MatchRequest(BaseModel):
    sheet_id:         str
    exclude_poem_ids: list[str] = []  # poem IDs already shown this session

class MusicFeatures(BaseModel):
    tempo_feel: str
    key_feel:   str
    mood:       str
    tags:       str

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


### API endpoints 

@app.get("/sheets")
def get_sheets():
    """
    Return all music pieces for the frontend dropdown.
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
    """
    Given a sheet_id and optional list of already-shown poem IDs:
    1. Look up pre-computed music tags from sheets_metadata.json
    2. Query ChromaDB RAG — skipping any excluded poem IDs
    3. Return music features + top 2 matched poems
    """
    sheets = load_music_sheets()

    sheet = next((s for s in sheets if s["id"] == request.sheet_id), None)
    if not sheet:
        raise HTTPException(
            status_code=404,
            detail=f"Sheet '{request.sheet_id}' not found."
        )

    # pass exclude_poem_ids to RAG 
    poems = query_poems(
        tags=sheet["tags"],
        exclude_ids=request.exclude_poem_ids # skip alr shown ones
    )

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


@app.get("/health")
def health_check():
    return {"status": "ok"}