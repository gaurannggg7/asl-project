import re, nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.data.path.append("/Users/gaurangmohan/nltk_data")

nltk.download('punkt', quiet=True)
STOPWORDS = {"is", "am", "are", "the", "a", "an", "to"}

def english_to_asl_gloss(text: str) -> list[str]:
    """Very rough rule-based reordering: Time-Topic-Comment."""
    glosses = []
    for s in sent_tokenize(text):
        words = [w.lower() for w in word_tokenize(s) if w.isalnum()]
        # heuristic: move day/time expressions to front
        time_words = [w for w in words if re.fullmatch(r"\d{1,2}(:\d{2})?(am|pm)?", w)]
        content = [w for w in words if w not in time_words and w not in STOPWORDS]
        glosses.extend(time_words + content)
    return [g.upper() for g in glosses]
