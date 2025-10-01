"""
Context Preparation Module for Math Problem Solver
Formats parsed paper content into structured context for LLM consumption.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from arxiv_fetcher import ArxivPaper
from latex_parser import LatexParser, TheoremStatement, ContentType
import os, json, hashlib, re


@dataclass
class FormattedContext:
    """Structured context for LLM consumption."""
    paper_metadata: Dict
    paper_context: str
    theorem_to_prove: TheoremStatement
    available_theorems: List[TheoremStatement]
    strategy_summary: Optional[str] = None


class ContextPreparator:
    """Prepares structured context from parsed mathematical papers."""
    
    def __init__(self, max_context_length: int = 50000):
        """
        Initialize context preparator.
        
        Args:
            max_context_length: Maximum character length for context
        """
        self.max_context_length = max_context_length
        self.parser = LatexParser()
        # Will hold provenance information for last prepared context
        self._last_provenance: Dict = {}
    
    def prepare_context(self, paper: ArxivPaper, target_theorem_index: int = 0,
                        rag_enabled: bool = False,
                        rag_chunk_size: int = 900,
                        rag_overlap: int = 150,
                        rag_top_k: int = 8,
                        small_doc_factor: float = 1.5,
                        rag_cache_dir: str = 'data/rag_cache',
                        force_rag: bool = False) -> FormattedContext:
        """
        Prepare structured context for proving a specific theorem.
        
        Args:
            paper: ArxivPaper object with LaTeX content
            target_theorem_index: Index of theorem to prove (0-based)
            rag_enabled: Whether to enable retrieval augmentation
            rag_chunk_size: Character size per chunk for RAG
            rag_overlap: Overlap size between chunks
            rag_top_k: Top-k retrieved chunks to include
            
        Returns:
            FormattedContext object ready for LLM
        """
        # Parse the paper
        parsed = self.parser.parse(paper.latex_content)

        # Remove proofs from content (used for both full-context and RAG)
        content_without_proofs = self.parser.remove_proofs(paper.latex_content)
        
        # Get target theorem
        theorems = parsed['theorems']
        if not theorems or target_theorem_index >= len(theorems):
            raise ValueError(f"No theorem at index {target_theorem_index}")
        
        target_theorem = theorems[target_theorem_index]
        
        # Get other available theorems (excluding the target)
        available_theorems = [thm for i, thm in enumerate(theorems) if i != target_theorem_index]
        
        # Create paper metadata
        metadata = {
            'arxiv_id': paper.arxiv_id,
            'title': paper.title,
            'authors': paper.authors,
            'subjects': paper.subjects,
            'published_date': paper.published_date.isoformat()
        }
        
        # Reset provenance container
        self._last_provenance = {
            'sections': [],  # list of {title, original_chars, included_chars, truncated}
            'available_theorems_count': 0,
            'target_theorem_index': target_theorem_index,
            'truncated_context': False,
            'context_char_limit': self.max_context_length
        }

        # Build structured context
        context_parts: List[str] = []
        
        # Add paper metadata
        context_parts.append(self._format_metadata(metadata))
        
        # Add abstract if available
        if parsed['abstract']:
            context_parts.append(self._format_abstract(parsed['abstract']))
        
        # Add introduction if available
        if parsed['introduction']:
            context_parts.append(self._format_introduction(parsed['introduction']))
        
        # Add key sections (excluding proofs) unless we use RAG (RAG will supply its own chunk block)
        sections_block = self._format_sections(parsed['sections'])
        if not rag_enabled:
            context_parts.append(sections_block)
        
        # If RAG enabled, build retrieval chunks (will insert BEFORE available theorems)
        rag_metadata: Optional[Dict] = None
        if rag_enabled:
            rag_metadata = {
                'enabled': True,
                'small_doc_bypass': False,
                'cache_used': False,
                'cache_path': None,
                'query_raw': None,
                'query_normalized': None,
                'force_rag': force_rag
            }
            # Small-document bypass check
            doc_chars = len(content_without_proofs)
            rag_metadata['doc_chars'] = doc_chars
            rag_metadata['chunk_size'] = rag_chunk_size
            rag_metadata['overlap'] = rag_overlap
            rag_metadata['top_k_requested'] = rag_top_k
            threshold = int(small_doc_factor * rag_chunk_size)
            rag_metadata['small_doc_threshold'] = threshold
            if doc_chars <= threshold and not force_rag:
                rag_metadata['enabled'] = False
                rag_metadata['small_doc_bypass'] = True
            else:
                    try:
                        from rag_chunker import chunk_text
                        from rag_index import LexicalIndex
                        # Prepare cache
                        os.makedirs(rag_cache_dir, exist_ok=True)
                        content_hash = hashlib.sha256(content_without_proofs.encode('utf-8')).hexdigest()
                        cache_path = os.path.join(rag_cache_dir, f"{paper.arxiv_id.replace('/', '_')}_{rag_chunk_size}_{rag_overlap}.json")
                        chunks = None
                        if os.path.exists(cache_path):
                            try:
                                with open(cache_path, 'r', encoding='utf-8') as cf:
                                    cached = json.load(cf)
                                if cached.get('hash') == content_hash:
                                    chunks = []
                                    for c in cached.get('chunks', []):
                                        from rag_chunker import Chunk
                                        chunks.append(Chunk(id=c['id'], text=c['text'], start_char=c['start_char'], end_char=c['end_char']))
                                    rag_metadata['cache_used'] = True
                                    rag_metadata['cache_path'] = cache_path
                                    rag_metadata['hash_prefix'] = content_hash[:12]
                                else:
                                    rag_metadata['cache_miss_reason'] = 'hash_mismatch'
                            except Exception as e:
                                rag_metadata['cache_miss_reason'] = f'cache_load_error: {e}'
                        if chunks is None:
                            # Chunk the proof-stripped content
                            chunks = chunk_text(content_without_proofs, chunk_size=rag_chunk_size, overlap=rag_overlap)
                            # Save cache
                            try:
                                with open(cache_path, 'w', encoding='utf-8') as cf:
                                    json.dump({
                                        'paper_id': paper.arxiv_id,
                                        'hash': content_hash,
                                        'chunk_size': rag_chunk_size,
                                        'overlap': rag_overlap,
                                        'chunks': [
                                            {'id': c.id, 'start_char': c.start_char, 'end_char': c.end_char, 'text': c.text}
                                            for c in chunks
                                        ]
                                    }, cf, ensure_ascii=False, indent=2)
                                rag_metadata['cache_path'] = cache_path
                                rag_metadata['hash_prefix'] = content_hash[:12]
                            except Exception as e:
                                rag_metadata['cache_miss_reason'] = f'cache_write_error: {e}'
                        idx = LexicalIndex()
                        idx.add_chunks([(c.id, c.text) for c in chunks])
                        # Query: use target theorem statement plus optional title
                        raw_query = target_theorem.statement
                        if target_theorem.title:
                            raw_query = f"{target_theorem.title}. {raw_query}"
                        rag_metadata['query_raw'] = raw_query
                        normalized_query = self._normalize_query(raw_query)
                        rag_metadata['query_normalized'] = normalized_query
                        query_for_scoring = normalized_query if normalized_query.strip() else raw_query
                        scored = idx.score(query_for_scoring, top_k=rag_top_k)
                        selected_chunks = []
                        for cid, score in scored:
                            chunk_obj = next((c for c in chunks if c.id == cid), None)
                            if not chunk_obj:
                                continue
                            selected_chunks.append((chunk_obj, score))

                        # ----- Recall Safety Net -----
                        try:
                            key_tokens = self._extract_key_tokens(raw_query)
                        except Exception as _tok_err:
                            key_tokens = []
                            rag_metadata['coverage_error'] = str(_tok_err)
                        covered = set()
                        if key_tokens:
                            for ch, _ in selected_chunks:
                                lower_txt = ch.text.lower()
                                for t in key_tokens:
                                    if t in lower_txt:
                                        covered.add(t)
                        coverage_before = (len(covered) / len(key_tokens)) if key_tokens else 1.0
                        rag_metadata['token_coverage_before'] = coverage_before
                        fallback_added = False
                        fallback_limit = 2
                        if key_tokens and coverage_before < 0.6:
                            missing = [t for t in key_tokens if t not in covered]
                            candidate_list = []
                            selected_ids = {c.id for c, _ in selected_chunks}
                            for c in chunks:
                                if c.id in selected_ids:
                                    continue
                                lower_txt = c.text.lower()
                                new_hits = [t for t in missing if t in lower_txt]
                                if new_hits:
                                    candidate_list.append((c, len(new_hits)))
                            candidate_list.sort(key=lambda x: (-x[1], len(x[0].text)))
                            for c, _score in candidate_list[:fallback_limit]:
                                synthetic = (min(s for _, s in selected_chunks) - 1e-6) if selected_chunks else 0.0
                                selected_chunks.append((c, synthetic))
                                fallback_added = True
                                lower_txt = c.text.lower()
                                for t in key_tokens:
                                    if t in lower_txt:
                                        covered.add(t)
                            rag_metadata['fallback_added'] = fallback_added
                            if fallback_added:
                                rag_metadata['fallback_chunk_ids'] = [c.id for c, _ in selected_chunks if c.id not in selected_ids]
                        else:
                            rag_metadata['fallback_added'] = False
                        coverage_after = (len(covered) / len(key_tokens)) if key_tokens else 1.0
                        rag_metadata['token_coverage_after'] = coverage_after

                        # ----- Merge Adjacent Chunks -----
                        merge_gap = 50
                        # Sort by start_char
                        selected_chunks.sort(key=lambda x: x[0].start_char)
                        merged_groups = []
                        merged_list = []  # (merged_chunk_like_object, combined_score)
                        from dataclasses import dataclass
                        @dataclass
                        class _Merged:
                            id: str
                            text: str
                            start_char: int
                            end_char: int
                        current_group = []
                        for ch, sc in selected_chunks:
                            if not current_group:
                                current_group.append((ch, sc))
                                continue
                            prev_ch, _prev_sc = current_group[-1]
                            if ch.start_char <= prev_ch.end_char + merge_gap:
                                current_group.append((ch, sc))
                            else:
                                # finalize current group
                                if len(current_group) == 1:
                                    only_ch, only_sc = current_group[0]
                                    merged_list.append((only_ch, only_sc))
                                    merged_groups.append([only_ch.id])
                                else:
                                    ids = [c.id for c, _ in current_group]
                                    text = " ".join(c.text for c, _ in current_group)
                                    start = current_group[0][0].start_char
                                    end = current_group[-1][0].end_char
                                    avg_score = sum(s for _, s in current_group) / len(current_group)
                                    merged_obj = _Merged(id=f"m_{ids[0]}_{ids[-1]}", text=text, start_char=start, end_char=end)
                                    merged_list.append((merged_obj, avg_score))
                                    merged_groups.append(ids)
                                current_group = [(ch, sc)]
                        # finalize last group
                        if current_group:
                            if len(current_group) == 1:
                                only_ch, only_sc = current_group[0]
                                merged_list.append((only_ch, only_sc))
                                merged_groups.append([only_ch.id])
                            else:
                                ids = [c.id for c, _ in current_group]
                                text = " ".join(c.text for c, _ in current_group)
                                start = current_group[0][0].start_char
                                end = current_group[-1][0].end_char
                                avg_score = sum(s for _, s in current_group) / len(current_group)
                                merged_obj = _Merged(id=f"m_{ids[0]}_{ids[-1]}", text=text, start_char=start, end_char=end)
                                merged_list.append((merged_obj, avg_score))
                                merged_groups.append(ids)

                        rag_metadata['merged'] = any(len(g) > 1 for g in merged_groups)
                        if rag_metadata['merged']:
                            rag_metadata['merged_groups'] = merged_groups
                        # Replace selected_chunks with merged_list for formatting
                        selected_chunks = merged_list
                        # Format retrieved chunks block
                        retrieved_block = ["# Retrieved Context Chunks (RAG)"]
                        if not selected_chunks:
                            retrieved_block.append("(No relevant chunks retrieved; falling back to truncated sections if available.)")
                            # If retrieval empty, include sections as fallback
                            context_parts.append(sections_block)
                            rag_metadata['retrieval_empty'] = True
                        else:
                            for rank, (ch, sc) in enumerate(selected_chunks, 1):
                                retrieved_block.append(f"**Chunk {rank} (id={ch.id}, score={sc:.4f})**: {ch.text}")
                            total_selected_chars = sum(len(ch.text) for ch, _ in selected_chunks)
                        rag_metadata['selected_coverage_ratio'] = (total_selected_chars / doc_chars) if doc_chars else 0.0
                        rag_metadata['effective_top_k'] = len(selected_chunks)
                        context_parts.append("\n".join(retrieved_block))
                        # Capture rag provenance
                        rag_metadata.update({
                            'total_chunks': len(chunks),
                            'top_k': rag_top_k,
                            'selected': [
                                {
                                    'id': ch.id,
                                    'score': sc,
                                    'start_char': ch.start_char,
                                    'end_char': ch.end_char,
                                    'char_len': len(ch.text)
                                } for ch, sc in selected_chunks
                            ]
                        })
                    except Exception as e:
                        # Retrieval attempt failed; keep enabled flag (if forced or not bypassed) but record error
                        context_parts.append("# Retrieved Context Chunks (RAG)\n[retrieval_error] falling back to sections.")
                        context_parts.append(sections_block)
                        if rag_metadata is None:
                            rag_metadata = {}
                        rag_metadata.update({
                            'retrieval_error': str(e)
                        })

        # If RAG enabled but small-doc bypass triggered, ensure sections included (already appended earlier)
        if rag_enabled and rag_metadata and rag_metadata.get('small_doc_bypass') and not rag_metadata.get('force_rag'):
            # Retrieval skipped due to small document; nothing further required
            pass

        # After retrieval (or if disabled), append available theorems and references
        if available_theorems:
            context_parts.append(self._format_available_theorems(available_theorems))
            self._last_provenance['available_theorems_count'] = len(available_theorems)
        if parsed['references']:
            context_parts.append(self._format_references(parsed['references']))

        # Combine all parts
        full_context = "\n\n".join(context_parts)

        # Truncate if too long
        if len(full_context) > self.max_context_length:
            full_context = self._truncate_context(full_context, context_parts)
            self._last_provenance['truncated_context'] = True

        # Generate strategy summary (placeholder for now)
        strategy = self._generate_strategy_summary(paper, target_theorem, parsed)

        # Attach RAG metadata into provenance if present
        if rag_metadata is not None:
            self._last_provenance['rag'] = rag_metadata
        else:
            # Explicitly note if not enabled
            self._last_provenance['rag'] = {'enabled': False}

        return FormattedContext(
            paper_metadata=metadata,
            paper_context=full_context,
            theorem_to_prove=target_theorem,
            available_theorems=available_theorems,
            strategy_summary=strategy
        )

    # ---------------- Internal Helpers ----------------
    def _extract_key_tokens(self, text: str) -> List[str]:
        """Extract salient tokens from theorem text for coverage heuristic.

        Steps:
          1. Remove inline LaTeX math delimiters $...$ and \( ... \)
          2. Strip LaTeX commands (\\alpha, \\sum, etc.)
          3. Lowercase and split on non-alphanumeric (keep vertical bar and digits briefly)
          4. Filter stopwords & very short tokens (<=2 chars) unless purely numeric
          5. Deduplicate preserving order; cap list size
        """
        import re
        if not text:
            return []
        # Remove common math environments inline
        t = re.sub(r"\$[^$]*\$", " ", text)  # $...$
        t = re.sub(r"\\\([^)]*\\\)", " ", t)  # \( ... \)
        # Remove LaTeX commands
        t = re.sub(r"\\[a-zA-Z]+", " ", t)
        # Normalize whitespace
        t = t.lower()
        # Keep alphanumerics and vertical bars (help preserve |a| patterns) temporarily
        t = re.sub(r"[^a-z0-9|]+", " ", t)
        raw = t.split()
        stop = {
            'the','then','let','there','exists','for','some','any','all','and','or','of','a','an','to','in','on','we','be','that','with','by','if','is','are','as','at','from','this','such','which','it','can'
        }
        tokens: List[str] = []
        for tok in raw:
            if tok in stop:
                continue
            # Collapse multiple bars (|3a| -> 3a by dropping bars)
            stripped = tok.strip('|')
            if not stripped:
                continue
            if len(stripped) <= 2 and not stripped.isdigit():
                # keep single/double digits but drop short non-numerics
                continue
            tokens.append(stripped)
        # Deduplicate preserving order
        seen = set()
        ordered: List[str] = []
        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)
        # Cap to 25 tokens for efficiency
        return ordered[:25]

    def get_last_provenance(self) -> Dict:
        """Return provenance metadata for the most recently prepared context."""
        return self._last_provenance
    
    def _format_metadata(self, metadata: Dict) -> str:
        """Format paper metadata."""
        return f"""# Paper Information
- **arXiv ID**: {metadata['arxiv_id']}
- **Title**: {metadata['title']}
- **Authors**: {', '.join(metadata['authors'])}
- **Subjects**: {', '.join(metadata['subjects'])}
- **Published**: {metadata['published_date'][:10]}"""
    
    def _format_abstract(self, abstract: str) -> str:
        """Format abstract section."""
        return f"""# Abstract
{abstract}"""
    
    def _format_introduction(self, introduction: str) -> str:
        """Format introduction section."""
        # Truncate introduction if very long
        if len(introduction) > 2000:
            introduction = introduction[:2000] + "..."
        
        return f"""# Introduction
{introduction}"""
    
    def _format_sections(self, sections: List) -> str:
        """Format paper sections."""
        if not sections:
            return ""
        
        formatted = "# Paper Sections\n"
        
        for section in sections:
            original_len = len(section.content)
            section_content = section.content
            truncated = False
            if len(section_content) > 1500:
                section_content = section_content[:1500] + "..."
                truncated = True
            self._last_provenance['sections'].append({
                'title': section.title,
                'original_chars': original_len,
                'included_chars': len(section_content),
                'truncated': truncated
            })
            formatted += f"\n## {section.title}\n{section_content}\n"
        
        return formatted
    
    def _format_available_theorems(self, theorems: List[TheoremStatement]) -> str:
        """Format available theorems and lemmas."""
        if not theorems:
            return ""
        
        formatted = "# Available Results (Theorems, Lemmas, etc.)\n"
        formatted += "The following results are available for use in proofs:\n\n"
        
        for i, thm in enumerate(theorems, 1):
            thm_type = thm.statement_type.value.title()
            title_part = f" ({thm.title})" if thm.title else ""
            
            formatted += f"**{thm_type} {i}{title_part}**: {thm.statement}\n\n"
        
        return formatted
    
    def _format_references(self, references: str) -> str:
        """Format references section."""
        if len(references) > 1000:
            references = references[:1000] + "..."
        
        return f"""# References
{references}"""

    # ----------------------- RAG Helpers -----------------------
    _LATEX_CMD_RE = re.compile(r"\\\\[a-zA-Z]+\*?(\[[^]]*\])?(\{[^}]*\})?")
    _MATH_ENV_RE = re.compile(r"\$+([^$]+)\$+")
    _BRACE_RE = re.compile(r"[{}]")
    _WS_RE = re.compile(r"\s+")

    def _normalize_query(self, text: str) -> str:
        """Normalize theorem/title text for lexical retrieval.
        Steps: remove LaTeX commands, strip math delimiters, remove braces, collapse whitespace, lowercase.
        Preserve alphanumerics and basic math symbols.
        """
        if not text:
            return ""
        # Remove LaTeX commands
        txt = self._LATEX_CMD_RE.sub(" ", text)
        # Strip math delimiters (keep inner)
        txt = self._MATH_ENV_RE.sub(lambda m: f" {m.group(1)} ", txt)
        # Remove braces
        txt = self._BRACE_RE.sub(" ", txt)
        # Collapse whitespace
        txt = self._WS_RE.sub(" ", txt)
        txt = txt.strip().lower()
        return txt
    
    def _truncate_context(self, full_context: str, context_parts: List[str]) -> str:
        """Truncate context intelligently to fit within limits."""
        # Priority order: metadata > abstract > available_theorems > introduction > sections > references
        essential_parts = context_parts[:3]  # metadata, abstract, introduction
        optional_parts = context_parts[3:]   # sections, theorems, references
        
        # Start with essential parts
        truncated = "\n\n".join(essential_parts)
        
        # Add optional parts if they fit
        for part in optional_parts:
            if len(truncated) + len(part) + 2 <= self.max_context_length:
                truncated += "\n\n" + part
            else:
                # Truncate this part to fit
                remaining_space = self.max_context_length - len(truncated) - 2
                if remaining_space > 100:  # Only add if meaningful space left
                    truncated += "\n\n" + part[:remaining_space] + "..."
                break
        
        return truncated
    
    def _generate_strategy_summary(self, paper: ArxivPaper, theorem: TheoremStatement, 
                                 parsed: Dict) -> Optional[str]:
        """Generate a strategy summary for proving the theorem (placeholder)."""
        # This is a placeholder for future implementation
        # Could use LLM to generate strategy based on paper content
        
        strategy_hints = []
        
        # Basic heuristics based on theorem type
        if theorem.statement_type == ContentType.THEOREM:
            strategy_hints.append("This is a main theorem - likely requires combining multiple lemmas.")
        elif theorem.statement_type == ContentType.LEMMA:
            strategy_hints.append("This is a lemma - may be a stepping stone to larger results.")
        
        # Check for common mathematical structures
        statement_lower = theorem.statement.lower()
        if 'infinite' in statement_lower:
            strategy_hints.append("Statement involves infinity - consider proof by contradiction or induction.")
        if 'for all' in statement_lower or 'every' in statement_lower:
            strategy_hints.append("Universal statement - consider direct proof or contradiction.")
        if 'exists' in statement_lower or 'there is' in statement_lower:
            strategy_hints.append("Existential statement - consider construction or contradiction.")
        
        if strategy_hints:
            return "Strategy hints: " + " ".join(strategy_hints)
        
        return None
    
    def format_for_llm_prompt(self, context: FormattedContext) -> str:
        """Format the context as a complete prompt for the LLM."""
        
        target_thm = context.theorem_to_prove
        thm_type = target_thm.statement_type.value.title()
        title_part = f" ({target_thm.title})" if target_thm.title else ""
        
        prompt = f"""You are an expert mathematician. Provide the best possible rigorous reasoning toward a proof of the following theorem.

If a fully rigorous proof is realistically out of reach with the given context, DO NOT refuse or say you cannot prove it. Instead:
- Still produce a Proof Sketch capturing the main ideas or plausible strategic avenues.
- In the Full Proof section, write as much rigorously justified argument as you can.
- For any steps you cannot currently justify, insert a clearly labeled gap line using the format:
  GAP: <concise description of missing lemma / estimate / classification still needed>
- Keep going after a gap if later steps plausibly depend on standard results; explicitly reference gaps when used.
- Prefer a small number of explicit GAP lines over vague handwaving.
- Never output only a disclaimer; always give constructive partial progress.

{context.paper_context}

---

# Theorem to Prove

**{thm_type}{title_part}**: {target_thm.statement}

---

# Task

Produce TWO components in order:

1. Proof Sketch (5-10 sentences): A high-level intuitive roadmap highlighting the key ideas, reductions, generating function manipulations, or combinatorial correspondences. Do NOT include full technical details here; focus on strategy and why it should work.
2. Full Detailed Proof: A fully rigorous derivation with explicit justifications, clear transitions, and concluding symbol (∎). Reference any prior results by their labels or descriptions if they appear in Available Results.

Guidelines:
- Do not merely restate the theorem; advance the argument logically.
- Prefer natural language sentences interwoven with math; avoid long walls of raw formulas.
- Explicitly identify crucial identities, injections/bijections, or generating function transformations before applying them.
- If multiple cases exist, label them Case 1, Case 2, etc.
- If an assumption is needed but not explicitly stated in the context, state it and justify its standardness.
- End the full proof with a standalone conclusion symbol (∎) on its own line or at the end of the final sentence.
- Avoid hallucinating external theorems not standard in the field unless you justify them as classical.

Output Format (Markdown headings):
### Proof Sketch
<concise high-level sketch>

### Proof
<full rigorous or best-effort partial proof with explicit GAP lines where necessary, ending with ∎ if you can legitimately conclude. If genuine uncertainty remains, you may end without ∎ but only after exhausting concrete partial derivations>

"""
        
        if context.strategy_summary:
            prompt += f"**Strategy Hint**: {context.strategy_summary}\n\n"
        
        prompt += "**Proof**:"
        
        return prompt


def main():
    """Test the context preparation module."""
    from arxiv_fetcher import ArxivFetcher
    from datetime import datetime
    
    # Fetch a paper
    fetcher = ArxivFetcher()
    papers = fetcher.search_papers(['math.NT'], datetime(2024, 8, 1), max_results=1)
    
    if papers and fetcher.download_latex_source(papers[0]):
        paper = papers[0]
        print(f"Preparing context for: {paper.title}")
        
        # Prepare context
        preparator = ContextPreparator()
        context = preparator.prepare_context(paper, target_theorem_index=0)
        
        print(f"\nPaper metadata: {context.paper_metadata['title']}")
        print(f"Context length: {len(context.paper_context)} characters")
        print(f"Target theorem: {context.theorem_to_prove.statement_type.value}")
        print(f"Available theorems: {len(context.available_theorems)}")
        
        # Show the complete LLM prompt
        prompt = preparator.format_for_llm_prompt(context)
        print(f"\nComplete prompt length: {len(prompt)} characters")
        print(f"\nFirst 500 characters of prompt:")
        print(prompt[:500] + "...")
        
    else:
        print("Failed to fetch paper")


if __name__ == "__main__":
    main()