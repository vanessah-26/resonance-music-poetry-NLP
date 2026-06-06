import sys
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

POEMS_FILE      = "poems.json"                
SHEETS_FILE     = "sheets/sheets_metadata.json"   
CHROMA_PATH     = "./chroma_db"                
COLLECTION_NAME = "poems"
TOP_K           = 2
CANDIDATE_POOL  = 10 


embedding_sentence = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH) 


def _get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_sentence,
        metadata={"hnsw:space": "cosine"}
    )


def poems_to_chroma_db() -> None:
    collection = _get_collection()
 
    if collection.count() > 0:
        print(f"ChromaDB already loaded ({collection.count()} poems). Skipping.")
        return
 
    with open(POEMS_FILE, "r", encoding="utf-8") as f:
        poems = json.load(f)
 
    print(f"Loading {len(poems)} poems into ChromaDB...")
 
    ids       = []
    documents = []  # for embedding tags
    metadatas = []  # stored as-is for display
 
    for poem in poems:
        ids.append(poem["id"])
 
        # Only embed tags (appended to the poem, see auto_tag.py) 
        documents.append(poem["tags"])
 
        metadatas.append({
            "title":       poem["title"],
            "poet":        poem["poet"],
            "language":    poem["language"],
            "original":    poem["original"],
            "translation": poem.get("translation") or "",
            "is_excerpt":  str(poem.get("is_excerpt", False)),
            "tags":        poem["tags"]
        })
 
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
 
    print(f"Loaded {collection.count()} poems into ChromaDB.")
 


def query_poems(tags: str, exclude_ids: list[str] = []) -> list[dict]:
    """ 
    RAG query steps:
    1. Embed music tags with the same model as the poem indexing step
    2. Get the candidate pool of 10 closest poems by cos similarity
    3. Skip poems in exclude_ids 
    4. Return top 2 matches at a time 
    """
    collection = _get_collection()
 
    if collection.count() == 0:
        raise RuntimeError("ChromaDB is empty. Call poems_to_chroma_db() first.")
 
    # Fetch the candidiate pool and exclude shown poems 
    n_candidates = min(
        CANDIDATE_POOL + len(exclude_ids),
        collection.count()
    )
 
    # Embed music tags and find closest poem vectors
    results = collection.query(
        query_texts=[tags],
        n_results=n_candidates
    )
 
    poems = []
    for i in range(len(results["ids"][0])):
        pid      = results["ids"][0][i]
        meta     = results["metadatas"][0][i]
        distance = results["distances"][0][i]
 
        # Skip poems already shown this session
        if pid in exclude_ids:
            continue
 
        poems.append({
            "id":               pid,
            "title":            meta["title"],
            "poet":             meta["poet"],
            "language":         meta["language"],
            "original":         meta["original"],
            "translation":      meta["translation"] if meta["translation"] else None,
            "is_excerpt":       meta["is_excerpt"] == "True",
            "tags":             meta["tags"],
            "similarity_score": round(1 - distance, 3)
        })
 
        if len(poems) == TOP_K:
            break
 
    return poems

def reset_db() -> None:
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Deleted existing ChromaDB collection.")
    except Exception:
        pass
    poems_to_chroma_db()
    print("ChromaDB reset and reloaded.")

#### Test RAG locally 

if __name__ == "__main__":
    with open(SHEETS_FILE, "r", encoding="utf-8") as f:
        sheets = json.load(f)
 
    print("\nAvailable music pieces:")
    for i, sheet in enumerate(sheets, 1):
        print(f"  {i:2}. {sheet['title']} -- {sheet['composer']}")
 
    try:
        choice = int(input("\nEnter number to test: ")) - 1
        if choice < 0 or choice >= len(sheets):
            print("Invalid choice.")
            sys.exit(1)
    except ValueError:
        print("Please enter a number.")
        sys.exit(1)
 
    selected = sheets[choice]
    print(f"\nSelected: {selected['title']} -- {selected['composer']}")
    print(f"Tags:     {selected['tags']}")
 
    print("\nInitializing ChromaDB...")
    poems_to_chroma_db()
 
    # Test with no exclusions first
    print(f"\nTop {TOP_K} matching poems (no exclusions):")
    results = query_poems(selected["tags"])
    shown_ids = []
    for i, poem in enumerate(results, 1):
        print(f"\n--- Match {i} (similarity: {poem['similarity_score']}) ---")
        print(f"Title:    {poem['title']}")
        print(f"Poet:     {poem['poet']}")
        print(f"Tags:     {poem['tags']}")
        shown_ids.append(poem["id"])
 
    # Test exclusion: simulate user selecting a second piece
    print(f"\n--- Simulating exclusion of: {shown_ids} ---")
    results2 = query_poems(selected["tags"], exclude_ids=shown_ids)
    print(f"\nTop {TOP_K} matching poems (with exclusions):")
    for i, poem in enumerate(results2, 1):
        print(f"\n--- Match {i} (similarity: {poem['similarity_score']}) ---")
        print(f"Title:    {poem['title']}")
        print(f"Poet:     {poem['poet']}")
        print(f"Tags:     {poem['tags']}")