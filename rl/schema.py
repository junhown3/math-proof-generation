"""Schema definitions for RLAIF pipeline.

This module defines lightweight data classes / helper constructors for:
- Candidate proof entries
- Judge scoring outputs (scalar rubric scores)
- Aggregated reward rows

We avoid heavy dependencies (e.g., pydantic) to keep the pipeline simple.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
import hashlib

RUBRIC_DIMENSIONS = [
    "correctness",        # Logical validity / absence of false claims
    "completeness",       # Coverage of necessary steps relative to canonical proof
    "cohesion",           # Flow / clarity / structural organization
    "retrieval_grounding",# Degree to which cited concepts appear in provided context
    "conciseness"         # Avoiding unnecessary verbosity
]

@dataclass
class CandidateMeta:
    paper_id: str
    theorem_index: int
    variant: str              # e.g. 'baseline', 'rag'
    model: str
    temperature: float
    seed: int
    rag_enabled: bool
    rag_top_k: int
    rag_chunk_size: int
    rag_overlap: int
    prompt_char_len: int
    context_char_len: int
    provenance: Optional[Dict[str, Any]] = None

@dataclass
class Candidate:
    candidate_id: str
    meta: CandidateMeta
    theorem_statement: str
    generated_proof: str

    @staticmethod
    def make_id(paper_id: str, theorem_index: int, variant: str, seed: int, ordinal: int) -> str:
        base = f"{paper_id}::{theorem_index}::{variant}::{seed}::{ordinal}"
        return hashlib.sha256(base.encode()).hexdigest()[:16]

    def to_json(self) -> str:
        return json.dumps({
            "candidate_id": self.candidate_id,
            "meta": asdict(self.meta),
            "theorem_statement": self.theorem_statement,
            "generated_proof": self.generated_proof
        }, ensure_ascii=False)

@dataclass
class JudgeScore:
    candidate_id: str
    paper_id: str
    theorem_index: int
    judge_model: str
    scores: Dict[str, float]  # dimension -> 1-10
    overall: float            # aggregated scalar
    rationale: Optional[str] = None
    canonical_used: bool = False
    prompt_hash: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

@dataclass
class RewardRow:
    candidate_id: str
    paper_id: str
    theorem_index: int
    reward: float
    model: str
    variant: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def aggregate_rubric(scores: Dict[str, float]) -> float:
    """Default aggregation: weighted mean (correctness/completeness heavier)."""
    weights = {
        "correctness": 0.35,
        "completeness": 0.25,
        "cohesion": 0.15,
        "retrieval_grounding": 0.15,
        "conciseness": 0.10
    }
    total = 0.0
    wsum = 0.0
    for k, v in scores.items():
        w = weights.get(k, 0.0)
        total += w * v
        wsum += w
    return total / wsum if wsum else 0.0

__all__ = [
    "CandidateMeta", "Candidate", "JudgeScore", "RewardRow", "RUBRIC_DIMENSIONS", "aggregate_rubric"
]
