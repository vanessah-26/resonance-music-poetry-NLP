import sys
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

POEMS_FILE      = "poems.json"                
SHEETS_FILE     = "sheets/sheets_metadata.json"   
CHROMA_PATH     = "./chroma_db"                
COLLECTION_NAME = "poems"
TOP_K           = 2


embedding_sentence = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH) 

def poems_to_chroma_db(): 
    """
    Load poems.json into ChromaDB  

    
    """

    # Create poems database and set up embedding function 
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_sentence,
        metadata={"hnsw:space": "cosine"}
    )
 
    # Check existed poems, skip if existed
    if collection.count() > 0:
        print(f"ChromaDB already loaded ({collection.count()} poems). Skipping.") 
        return
 
    with open(POEMS_FILE, "r", encoding="utf-8") as f:
        poems = json.load(f)
 
    print(f"Loading {len(poems)} poems into ChromaDB...")
 
    ids       = []
    documents = []  # for tags embedding
    metadatas = []  # stored as-is for display
 
    for poem in poems:
        ids.append(poem["id"])

        # only embed tags 
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
 


def query_poems(tags: str) -> list[dict]:

    """
    RAG retrieval: 
    1. Take a music piece's tag string, embed it with the same model as the poems 
    2. Find top K most similar poems based on tag embeddings that are closest in vector space(cosine similarity)
    3. Return the top K poems with metadata 

    """

    # Double check the same collection. 
    # If not exist, it means the DB is not initialized, raise error.

    collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_sentence,
    metadata={"hnsw:space": "cosine"}
    )
 
    if collection.count() == 0:
        raise RuntimeError("ChromaDB is empty. Call init_db() first.")
 

    # Retrieve top K poems 
    results = collection.query(
        query_texts=[tags],
        n_results=TOP_K
    )
 
    poems = []


    for i in range(len(results["ids"][0])):
        meta     = results["metadatas"][0][i]
        distance = results["distances"][0][i]
 
        poems.append({
            "id":               results["ids"][0][i],
            "title":            meta["title"],
            "poet":             meta["poet"],
            "language":         meta["language"],
            "original":         meta["original"],
            "translation":      meta["translation"] if meta["translation"] else None,
            "is_excerpt":       meta["is_excerpt"] == "True",
            "tags":             meta["tags"],
            "similarity_score": round(1 - distance, 3)
        })
 
    return poems

def reset_db():
    """
    Use when need to clear existing ChromaDB and reload from poems.json 
    (e.g. after updating poems).
    """
    try: 
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("Deleted existing ChromaDB collection.")
    except Exception:
        # print(f"Error resetting ChromaDB: {e}")
        pass 
    poems_to_chroma_db()
    print("ChromaDB reset and reloaded with poems.")


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
 
    results = query_poems(selected["tags"])
 
    print(f"\nTop {TOP_K} matching poems:")
    for i, poem in enumerate(results, 1):
        print(f"\n--- Match {i} (similarity: {poem['similarity_score']}) ---")
        print(f"Title:    {poem['title']}")
        print(f"Poet:     {poem['poet']}")
        print(f"Language: {poem['language']}")
        print(f"Tags:     {poem['tags']}")
        print(f"Excerpt:  {poem['is_excerpt']}")
        print(f"\n{poem['original']}")
        if poem["translation"]:
            print(f"\n[Translation]\n{poem['translation']}")