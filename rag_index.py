"""Simple lexical retrieval (TF-IDF-like scoring without heavy dependencies).
Builds an inverted index with document frequencies and scores chunks at query time.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import math
import re
from collections import defaultdict, Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|\\$[A-Za-z]+|\\\w+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]

@dataclass
class IndexedChunk:
    chunk_id: int
    text: str
    length: int

class LexicalIndex:
    def __init__(self):
        self.df: Dict[str, int] = defaultdict(int)
        self.postings: Dict[str, List[Tuple[int,int]]] = defaultdict(list)  # token -> list of (chunk_id, freq)
        self.chunks: Dict[int, IndexedChunk] = {}
        self.total_chunks = 0

    def add_chunks(self, chunks: List[Tuple[int,str]]):
        for cid, text in chunks:
            toks = tokenize(text)
            if not toks:
                continue
            counts = Counter(toks)
            for tok, freq in counts.items():
                self.postings[tok].append((cid, freq))
                self.df[tok] += 1
            self.chunks[cid] = IndexedChunk(chunk_id=cid, text=text, length=len(toks))
            self.total_chunks += 1

    def score(self, query: str, top_k: int = 8) -> List[Tuple[int,float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        # Build query term weights (raw freq)
        q_counts = Counter(q_tokens)
        cand_scores: Dict[int, float] = defaultdict(float)
        for tok, qf in q_counts.items():
            df = self.df.get(tok)
            if not df:
                continue
            idf = math.log(1 + (self.total_chunks / df))
            for cid, freq in self.postings[tok]:
                # simple tf * idf * query_freq
                tf = freq / (self.chunks[cid].length or 1)
                cand_scores[cid] += (tf * idf * qf)
        ranked = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

if __name__ == "__main__":
    from rag_chunker import chunk_text
    text = "The quick brown fox jumps over the lazy dog. The dog was not amused by the quick fox."  # noqa
    chunks = chunk_text(text, 40, 10)
    idx = LexicalIndex()
    idx.add_chunks([(c.id, c.text) for c in chunks])
    print(idx.score("quick dog"))
