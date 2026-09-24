import re
from collections import Counter


def clean_text(text: str) -> str:
    """Normalize text for simple NLP helpers."""
    if not text:
        return ""
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", str(text).lower())
    cleaned = re.sub(r"[^a-z0-9%₹\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def analyze_sentiment(text: str) -> dict:
    """Return a simple positive/negative/neutral sentiment label based on keywords."""
    cleaned = clean_text(text)
    positive_words = {"growth", "profit", "strong", "record", "increase", "gain", "positive", "improve"}
    negative_words = {"decline", "loss", "drop", "weak", "negative", "risk", "fall", "major"}
    score = 0
    for word in positive_words:
        score += cleaned.count(word)
    for word in negative_words:
        score -= cleaned.count(word)
    if score > 0:
        label = "Positive"
    elif score < 0:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "score": score}


def extract_entities(text: str) -> dict:
    """Extract simple named entities used in lightweight NLP tests."""
    cleaned = clean_text(text)
    entities = {"PERCENT": re.findall(r"\d+(?:\.\d+)?%", cleaned), "MONEY": re.findall(r"₹?\d+(?:,\d+)*(?:\.\d+)?(?: crore| lakh| million)?", cleaned)}
    return entities


def top_keywords(text: str, n: int = 5):
    """Return the most common keywords as a list of (word, count) tuples."""
    cleaned = clean_text(text)
    words = [word for word in cleaned.split() if len(word) > 2 and word not in {"this", "that", "with", "from"}]
    counts = Counter(words)
    return counts.most_common(n)
