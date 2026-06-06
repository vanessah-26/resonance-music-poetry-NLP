import json
from keybert import KeyBERT

POEMS_FILE  = "poems.json"
OUTPUT_FILE = "poems.json"       
MODEL_NAME  = "all-MiniLM-L6-v2"  # same model
TOP_KEYWORDS = 10               #5  #extract more, then map to mood vocab


# Mood vocab: maps raw extracted keywords to standardized mood tags

MOOD_MAP = {
    # Nature / setting
    "moon": "moonlit", "moonlight": "moonlit", "moonlit": "moonlit",
    "night": "nocturnal", "dark": "dark", "darkness": "dark",
    "sun": "bright", "light": "light", "bright": "bright",
    "star": "ethereal", "stars": "ethereal", "sky": "ethereal",
    "water": "flowing", "river": "flowing", "sea": "flowing", "ocean": "flowing",
    "rain": "melancholic", "storm": "stormy", "wind": "flowing",
    "snow": "cold", "frost": "cold", "winter": "cold",
    "spring": "hopeful", "flower": "delicate", "flowers": "delicate",
    "tree": "nature", "forest": "nature", "leaf": "nature", "leaves": "nature",
    "bird": "gentle", "eagle": "powerful", "tiger": "fierce",
    "mountain": "majestic", "field": "serene", "garden": "serene",

    # Emotion / mood
    "love": "romantic", "heart": "passionate", "soul": "devotional",
    "dream": "dreamy", "dreams": "hopeful", "hope": "hopeful",
    "grief": "sorrowful", "sorrow": "sorrowful", "sad": "melancholic",
    "sadness": "melancholic", "melancholy": "melancholic",
    "joy": "joyful", "happy": "joyful", "happiness": "joyful",
    "longing": "longing", "yearning": "longing", "miss": "longing",
    "fear": "tense", "terror": "dark", "death": "somber",
    "dead": "somber", "dying": "somber", "grave": "somber",
    "peace": "peaceful", "quiet": "quiet", "silence": "quiet",
    "war": "dramatic", "battle": "dramatic", "fight": "fierce",
    "glory": "triumphant", "victory": "triumphant", "triumph": "triumphant",
    "lost": "melancholic", "forgotten": "nostalgic", "memory": "nostalgic",
    "past": "nostalgic", "old": "nostalgic", "ancient": "grounded",
    "anger": "intense", "rage": "fierce", "passion": "passionate",
    "wonder": "ethereal", "beauty": "romantic", "beautiful": "romantic",

    # Rhythm / pace indicators
    "slowly": "slow", "still": "still", "quiet": "quiet", "silence": "quiet",
    "swift": "fast", "quick": "fast", "rushing": "fast", "fast": "fast",
    "gentle": "gentle", "soft": "gentle", "tender": "tender",
    "wild": "fierce", "fierce": "fierce", "bold": "bold",
    "deep": "contemplative", "vast": "majestic", "eternal": "timeless",

    # Imagery
    "golden": "warm", "gold": "warm", "silver": "ethereal",
    "red": "passionate", "blood": "dramatic", "fire": "fierce",
    "ice": "cold", "stone": "grounded", "iron": "grounded",
    "heaven": "ethereal", "god": "devotional", "divine": "devotional",
    "fate": "fateful", "time": "contemplative", "journey": "wandering",
    "road": "wandering", "path": "wandering", "sea": "flowing",
    "captain": "heroic", "ship": "dramatic", "bell": "solemn",
    "music": "lyrical", "song": "lyrical", "violin": "lyrical",
    "autumn": "autumnal", "summer": "warm", "spring": "hopeful",
    "laugh": "joyful", "smile": "gentle", "tear": "sorrowful",
    "tears": "sorrowful", "weep": "sorrowful", "cry": "sorrowful",
}

# Fallback tags when keyword mapping produces niot enough results
FALLBACK_TAGS = ["contemplative", "lyrical", "flowing", "gentle", "reflective"]


def get_poem_text(poem: dict) -> str:
    """
    - Get the best text for keyword extraction
    - For non-English poems, use the English translation
    - For English poems, use the original
    """
    if poem.get("translation"):
        return poem["translation"]
    return poem["original"]


def extract_tags(kw_model: KeyBERT, poem: dict) -> str:
    """
    Extract mood tags from poem text using KeyBERT.
    Returns a comma-separated string of 6 tags.
    """
    text = get_poem_text(poem)

    # Extract top keywords from poem text
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),   #single words only
        stop_words="english",
        top_n=TOP_KEYWORDS,
        use_mmr=True,                    #max marginal relevance for diversity
        diversity=0.5                    #balance relevance and diversity
    )

    # Map extracted keywords to mood vocab 
    mood_tags = []
    seen = set()

    for keyword, score in keywords:
        word = keyword.lower().strip()
        if word in MOOD_MAP:
            tag = MOOD_MAP[word]
            if tag not in seen:
                mood_tags.append(tag)
                seen.add(tag)

    # If didn't get enough tags from mapping, add relevant fallbacks based on the highest scoring raw keywords
    if len(mood_tags) < 6:
        for keyword, score in keywords:
            word = keyword.lower().strip()
            if word not in seen and len(word) > 3:
                mood_tags.append(word)
                seen.add(word)
            if len(mood_tags) >= 6:
                break

    # If still not enough, pad with fallbacks
    for fallback in FALLBACK_TAGS:
        if len(mood_tags) >= 6:
            break
        if fallback not in seen:
            mood_tags.append(fallback)
            seen.add(fallback)

    return ", ".join(mood_tags[:6])


def main():
    print("Loading poems...")
    with open(POEMS_FILE, "r", encoding="utf-8") as f:
        poems = json.load(f)

    print(f"Loaded {len(poems)} poems.")
    print(f"Loading KeyBERT with model: {MODEL_NAME}...")

    # Use the same model as RAG 
    kw_model = KeyBERT(model=MODEL_NAME)
    print("Extracting tags from poem text...\n")

    for poem in poems:
        old_tags = poem.get("tags", "")
        new_tags = extract_tags(kw_model, poem)
        poem["tags"] = new_tags

        print(f"{poem['id']} — {poem['title'][:45]}")
        print(f"  old: {old_tags}")
        print(f"  new: {new_tags}")
        print()

    # Save updated poems
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(poems, f, indent=2, ensure_ascii=False)

    print(f"Done. Updated {len(poems)} poems saved to {OUTPUT_FILE}")
    print()
    print("Next steps:")
    print("  1. Review the new tags above")
    print("  2. Push updated poems.json to your HF Space")
    print("  3. Reset ChromaDB: python -c \"from rag import reset_db; reset_db()\"")


if __name__ == "__main__":
    main()