"""RAG Chunker
Splits paper content into overlapping semantic-ish chunks for retrieval.
Lightweight, no external deps.
"""
from dataclasses import dataclass
from typing import List, Tuple
import re

@dataclass
class Chunk:
    id: int
    text: str
    start_char: int
    end_char: int

def clean_text(text: str) -> str:
    # Light cleanup: collapse whitespace
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[Chunk]:
    text = clean_text(text)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)
    chunks: List[Chunk] = []
    start = 0
    cid = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_txt = text[start:end]
        # Try to avoid cutting mid-sentence (look back for period)
        if end < len(text):
            period_pos = chunk_txt.rfind('. ')
            if period_pos != -1 and period_pos > chunk_size * 0.6:
                # Adjust end near sentence boundary
                end = start + period_pos + 1
                chunk_txt = text[start:end]
        chunks.append(Chunk(id=cid, text=chunk_txt.strip(), start_char=start, end_char=end))
        cid += 1
        if end == len(text):
            break
        start = max(end - overlap, end) if overlap > 0 else end
    return chunks

if __name__ == "__main__":
    sample = """This is a short sample text. It has several sentences. We want to see how chunking works. The goal is to produce overlapping segments for retrieval augmented generation."""
    for c in chunk_text(sample, 50, 10):
        print(c)
