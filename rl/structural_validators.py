"""Structural validators for mathematical proof candidates.

Provides lightweight heuristics to detect:
- Undefined symbol usage (symbol appears in proof but not introduced earlier)
- Lemma reference mismatch (references to Lemma N not in context provenance if available)

These are heuristic and NOT formal verification.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List, Set


SYMBOL_PATTERN = re.compile(r"\\?[A-Za-z]{1,2}[0-9]?")  # crude heuristic for single/two-letter symbols
LEMMA_REF_PATTERN = re.compile(r"Lemma\s+([0-9]+)")


def extract_symbols(text: str) -> List[str]:
    # Exclude common English words that match pattern accidentally
    stop = {"In","If","We","As","By","It","Or","On","So","An","Is","At","To"}
    found = [m.group(0) for m in SYMBOL_PATTERN.finditer(text)]
    cleaned = []
    for s in found:
        if s in stop:
            continue
        # Ignore LaTeX commands like \in, \to
        if s.startswith('\\'):
            continue
        cleaned.append(s)
    return cleaned


def validate_symbols(proof: str, theorem: str) -> Dict[str, Any]:
    theorem_syms = set(extract_symbols(theorem)[:100])
    proof_syms = extract_symbols(proof)[:800]
    introduced: Set[str] = set()
    undefined: Set[str] = set()
    # Simple left-to-right scan: symbol is defined first time it appears in theorem or after phrase "Let <sym>" / "Fix <sym>"
    let_pattern = re.compile(r"\b(Let|Fix)\s+([A-Za-z]{1,2}[0-9]?)\b")
    # Pre-seed with theorem symbols
    introduced |= theorem_syms
    for line in proof.split('\n'):
        for m in let_pattern.finditer(line):
            introduced.add(m.group(2))
        for s in extract_symbols(line):
            if s not in introduced:
                undefined.add(s)
    return {
        'symbol_total': len(set(proof_syms)),
        'symbol_introduced': len(introduced),
        'symbol_undefined': sorted(undefined)[:20],
        'symbol_undefined_count': len(undefined)
    }


def validate_lemma_references(proof: str, provenance: Dict[str, Any] | None) -> Dict[str, Any]:
    refs = LEMMA_REF_PATTERN.findall(proof)
    if not refs:
        return {'lemma_ref_count': 0, 'lemma_missing': [], 'lemma_missing_count': 0}
    # If provenance contains selected chunk texts, attempt naïve presence check
    available_text = ''
    if provenance and 'rag' in provenance and provenance['rag'].get('selected_chunks'):
        for ch in provenance['rag']['selected_chunks']:
            available_text += '\n' + ch.get('text','')
    missing: Set[str] = set()
    for r in refs:
        # Look for the lemma number literally in available text or treat as missing
        pattern = re.compile(rf"Lemma\s+{re.escape(r)}\b")
        if available_text and not pattern.search(available_text):
            missing.add(r)
    return {
        'lemma_ref_count': len(refs),
        'lemma_missing': sorted(missing)[:20],
        'lemma_missing_count': len(missing)
    }


def run_structural_validators(proof: str, theorem: str, provenance: Dict[str, Any] | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out.update(validate_symbols(proof, theorem))
    out.update(validate_lemma_references(proof, provenance))
    # Simple hallucination heuristic: many undefined symbols OR many missing lemma references
    hallucination_risk = 0
    if out.get('symbol_undefined_count', 0) > 5:
        hallucination_risk += 1
    if out.get('lemma_missing_count', 0) > 2:
        hallucination_risk += 1
    out['hallucination_risk_level'] = ['low','medium','high'][min(hallucination_risk,2)]
    return out

__all__ = ['run_structural_validators']
