# Resonance: Music & Poetry RAG System
Select a piece of classical music. Discover the poems it echoes.

### Live demo: https://vanessah-26.github.io/resonance-music-poetry-NLP/
---
## Features:
- 21 classical music pieces with pre-tagged musical features
- 61 multilingual poems (English, Vietnamese, Chinese, French)
- Custom audio player 
- Musical features panel (tempo, key, mood, semantic tags)
- Output high-matched poems: no repeated poems music across selections
- Two new high-matched poems every “Find Matching Poems” click 

*Note*: The first top 2 poems is always the highest matching result. Re-running "Find Matching Poems" on the same music piece will get new matching poems. However, the new result of poems may overlap with other music pieces, and/or less relevant to the current piece.
---
## NLP Pipelines: 
```
Poem text (English or translation)
        ↓
KeyBERT: extracts top 10 keywords using SentenceTransformers embeddings
        ↓
MMR diversification (diversity=0.5): ensures most relevant keywords representing the poem without redundancy 
        ↓
MOOD_MAP: domain-specific lexicon maps raw words to mood vocabulary
        ↓
6 tags saved to poems.json
        ↓
SentenceTransformers embeds tags → 384-dim vectors stored in ChromaDB
        ↓
At query time: music tags embedded → cosine similarity search → top 2 poems returned
```
### RAG Implementation
**The first ever startup:**
Each poem's tag string is embedded by `all-MiniLM-L6-v2` (SentenceTransformers) into a 384-dimensional vector and stored in ChromaDB with poem metadata. Every startup after that, chroma_db/ already has data so it will skip and loads instantly.
  
**Retrieval (per query):**
1. Music tags are looked up from `sheets_metadata.json`
2. Tags are embedded using the same model
3. ChromaDB performs cosine similarity search across all 61 poem vectors
4. Poems in `exclude_poem_ids` (already shown this session) are skipped to ensure diverse poems output and prevent repetitive poems across different pieces.
5. Top 2 remaining matches are returned with similarity scores
The same embedding model is used for both music tags and poem tags, placing them in a shared semantic vector space where musical mood and poetic mood can be directly compared.

---
## Project Structure
```
music-poetry/                   
├── index.html                 # Frontend: Github page app
├── auto_tag.py                # NLP tagging script (run once locally)
├── app.py                     # FastAPI backend 
├── rag.py                     # RAG pipeline  
├── extract.py                 # Phase 2: extract from sheet music 
├── poems.json                 # 60 poems with auto-generated tags (see auto_tag.py)
├── requirements.txt
└── README.md

resonance-backend-clean/       ← Hugging Face Space (backend)
├── app.py
├── rag.py
├── extract.py
├── poems.json
├── requirements.txt
├── Dockerfile
└── sheets/
    ├── sheets_metadata.json   # 21 pieces with musical feature tags
    └── audio/                 # MP3 files (served from HF dataset: https://huggingface.co/datasets/tvhessa/music-poem-nlp2026, handcrafted collected from opensource website: https://www.classicals.de/)
```
---
## Setup locally
```bash
pip install fastapi uvicorn sentence-transformers chromadb python-dotenv aiofiles keybert
```
### Generate tags for poems (run once)
```bash
python auto_tag.py
```
This will read each poem (English original or translation if non-English texts), extracts keywords using KeyBERT. It converts each poem into a vector, then each word into individual vectors, and use MMR and MOOD_MAP to generate 6 tags that will automatically write to `poems.json`.

### Backend
```bash
uvicorn app:app --reload
```
At the start, `poems_to_chroma_db()` embeds all poem tags into ChromaDB. Subsequent startups skip this step if the DB is already populated.

### Frontend
Open `index.html` in your browser. The frontend calls `http://localhost:8000` by default.

### Reset ChromaDB after updating poems.json
```bash
python -c "from rag import reset_db; reset_db()"
```
--- 
### Deploy backend to Hugging Face
```bash
cd resonance-backend-clean
git add .
git commit -m "update"
git push hf main
```
---
## Poetry Corpus
 
61 poems across 4 languages:
 
| Language | Count | Poets include |
|---|---|---|
| English | 41 | Shakespeare, Dickinson, Frost, Keats, Blake, Whitman, Poe, Yeats, Hopkins, Kipling, Byron, Tennyson, Tupac Shakur |
| Vietnamese | 11 | Nguyễn Du, Xuân Diệu, Hàn Mặc Tử, Hồ Chí Minh, Tố Hữu, Bằng Việt, Xuân Quỳnh |
| Chinese | 4 | Li Bai, Yang Wanli, Wang Wei, Li Shen |
| French | 5 | Verlaine, Hugo, Rimbaud, Desbordes-Valmore, La Fontaine |
 
For non-English poems, `auto_tag.py` uses the English translation for keyword extraction to ensure alignment with the music tag vocabulary.
 
---
 
## Music Corpus
 
21 classical pieces spanning Baroque to Impressionist periods, covering a wide range of moods.
 
Each piece has pre-computed features:
- `tempo_feel` - slow / moderate / fast
- `key_feel` — major / minor
- `mood` — semantic mood descriptions
- `tags` — semantic tags used for RAG retrieval
- `audio_file` — URL to MP3 on HF dataset

---
## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript (vanilla) |
| Backend | Python, FastAPI, Uvicorn |
| Vector DB | ChromaDB |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Keyword extraction | KeyBERT + MMR |
| Containerization | Docker |
| Frontend hosting | GitHub Pages |
| Backend hosting | Hugging Face Spaces |
| Audio hosting | Hugging Face Datasets |
---
### Author: Vanessa Huynh