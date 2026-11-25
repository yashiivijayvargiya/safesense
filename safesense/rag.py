"""Tiny RAG over knowledge_base.json. Gemini embeddings if a key exists, else TF-IDF."""
import json, pathlib
import numpy as np
from . import llm

KB_PATH = pathlib.Path(__file__).parent.parent / "knowledge_base.json"

class Retriever:
    def __init__(self):
        self.docs = json.loads(KB_PATH.read_text(encoding="utf-8"))
        self.texts = [f"{d['title']}. {d['text']}" for d in self.docs]
        self.mode = "tfidf"
        if llm.available():
            try:
                self.vecs = llm.embed(self.texts)
                self.mode = "embeddings"
            except Exception:
                pass
        if self.mode == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.mat = self.tf.fit_transform(self.texts)

    def search(self, query: str, k: int = 5, min_score: float = 0.05):
        if self.mode == "embeddings":
            scores = self.vecs @ llm.embed([query])[0]
        else:
            scores = (self.mat @ self.tf.transform([query]).T).toarray().ravel()
        idx = np.argsort(-scores)[:k]
        return [{**self.docs[i], "score": float(scores[i])} for i in idx if scores[i] >= min_score]
