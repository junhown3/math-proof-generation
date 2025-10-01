"""
LaTeX Parser for Mathematical Papers
Extracts structure from LaTeX files: sections, theorems, proofs, references, etc.
"""

import re
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    """Types of content blocks in a mathematical paper."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    SECTION = "section"
    SUBSECTION = "subsection"
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    DEFINITION = "definition"
    REMARK = "remark"
    EXAMPLE = "example"
    PROOF = "proof"
    REFERENCES = "references"
    APPENDIX = "appendix"


@dataclass
class ContentBlock:
    """Represents a block of content in the paper."""
    content_type: ContentType
    title: Optional[str]
    content: str
    section_number: Optional[str] = None
    theorem_number: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass 
class TheoremStatement:
    """Represents a theorem-like statement with its proof (if present)."""
    statement_type: ContentType  # THEOREM, LEMMA, etc.
    number: Optional[str]
    title: Optional[str] 
    statement: str
    proof: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class LatexParser:
    """Parser for mathematical LaTeX papers."""
    
    # Common LaTeX environments for theorem-like statements
    THEOREM_ENVIRONMENTS = {
        'theorem': ContentType.THEOREM,
        'thm': ContentType.THEOREM,
        'lemma': ContentType.LEMMA,
        'lem': ContentType.LEMMA,
        'proposition': ContentType.PROPOSITION,
        'prop': ContentType.PROPOSITION,
        'corollary': ContentType.COROLLARY,
        'cor': ContentType.COROLLARY,
        'definition': ContentType.DEFINITION,
        'defn': ContentType.DEFINITION,
        'remark': ContentType.REMARK,
        'example': ContentType.EXAMPLE,
    }
    
    # Common proof environments
    PROOF_ENVIRONMENTS = ['proof', 'pf', 'dem', 'demonstration']
    
    def __init__(self):
        self.content_blocks: List[ContentBlock] = []
        self.theorems: List[TheoremStatement] = []
        
    def parse(self, latex_content: str) -> Dict:
        """
        Parse LaTeX content and extract structured information.
        
        Args:
            latex_content: Raw LaTeX content
            
        Returns:
            Dictionary with parsed content structure
        """
        self.content_blocks = []
        self.theorems = []
        
        # Clean and normalize content
        cleaned_content = self._clean_latex(latex_content)
        lines = cleaned_content.split('\n')
        
        # Extract different components
        abstract = self._extract_abstract(cleaned_content)
        introduction = self._extract_introduction(cleaned_content)
        sections = self._extract_sections(cleaned_content)
        theorems = self._extract_theorems(cleaned_content)
        references = self._extract_references(cleaned_content)
        
        return {
            'abstract': abstract,
            'introduction': introduction,
            'sections': sections,
            'theorems': theorems,
            'references': references,
            'full_content': cleaned_content
        }
    
    def _clean_latex(self, content: str) -> str:
        """Clean and normalize LaTeX content."""
        # Remove comments
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def _extract_abstract(self, content: str) -> Optional[str]:
        """Extract the abstract section."""
        # Look for abstract environment
        abstract_match = re.search(
            r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
            content, re.DOTALL | re.IGNORECASE
        )
        
        if abstract_match:
            return abstract_match.group(1).strip()
        
        return None
    
    def _extract_introduction(self, content: str) -> Optional[str]:
        """Extract the introduction section."""
        # Look for introduction section
        intro_patterns = [
            r'\\section\*?\{[Ii]ntroduction\}(.*?)(?=\\section|\Z)',
            r'\\section\*?\{\s*1\.?\s*[Ii]ntroduction\}(.*?)(?=\\section|\Z)'
        ]
        
        for pattern in intro_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_sections(self, content: str) -> List[ContentBlock]:
        """Extract all sections and subsections."""
        sections = []
        
        # Find all section-like commands
        section_pattern = r'\\(sub)?section\*?\{([^}]+)\}(.*?)(?=\\(?:sub)?section|\Z)'
        
        for match in re.finditer(section_pattern, content, re.DOTALL):
            is_subsection = match.group(1) == 'sub'
            title = match.group(2)
            section_content = match.group(3).strip()
            
            content_type = ContentType.SUBSECTION if is_subsection else ContentType.SECTION
            
            sections.append(ContentBlock(
                content_type=content_type,
                title=title,
                content=section_content
            ))
        
        return sections
    
    def _extract_theorems(self, content: str) -> List[TheoremStatement]:
        """Extract all theorem-like statements."""
        theorems = []
        
        # Pattern for theorem environments
        for env_name, content_type in self.THEOREM_ENVIRONMENTS.items():
            # Pattern: \begin{theorem}[optional title]{optional content} ... \end{theorem}
            pattern = rf'\\begin\{{{env_name}\}}(?:\[([^\]]*)\])?(.*?)\\end\{{{env_name}\}}'
            
            for match in re.finditer(pattern, content, re.DOTALL | re.IGNORECASE):
                title = match.group(1) if match.group(1) else None
                raw_statement = match.group(2).strip()

                # Normalize internal environments that often fragment prose
                statement = raw_statement
                # Replace description/itemize labels with simple markers preserving (a),(b), etc.
                # Convert \item{(a)} or \item[(a)] or \item (a) variants.
                statement = re.sub(r'\\item\s*\{?\(?([a-zA-Z0-9]+)\)?\}?', r' (\1) ', statement)
                statement = re.sub(r'\\item\s*\[(.*?)\]', r' (\1) ', statement)
                # Remove begin/end of description/itemize/enumerate but keep spacing
                statement = re.sub(r'\\begin\{(description|itemize|enumerate)\}', ' ', statement)
                statement = re.sub(r'\\end\{(description|itemize|enumerate)\}', ' ', statement)
                # Normalize display math blocks to placeholder tokens first
                statement = re.sub(r'\\begin\{equation\*?\}', ' <EQ> ', statement)
                statement = re.sub(r'\\end\{equation\*?\}', ' </EQ> ', statement)
                statement = re.sub(r'\\\\', ' ', statement)  # line breaks to space
                # Split out item parts for (a),(b),(c),(d) each on new line if present
                # We'll search for pattern (a) ... (b) ... etc. and structure them.
                def _split_items(text: str):
                    # Find markers like (a) (b) (c) ...
                    parts = re.split(r' (?=\([a-zA-Z0-9]+\) )', text)
                    # If only one part, return as is
                    if len(parts) == 1:
                        return text
                    # Rejoin with line breaks
                    rebuilt = []
                    for p in parts:
                        p = p.strip()
                        if not p:
                            continue
                        rebuilt.append(p)
                    return '\n'.join(rebuilt)
                statement = _split_items(statement)

                # Convert placeholder EQ tags to display math \[ ... \]
                def _convert_eq(line: str):
                    # Replace sequences <EQ> math </EQ>
                    # First collapse multiple spaces
                    line = re.sub(r'\s+', ' ', line)
                    # Replace '<EQ>' boundaries with markers; assume content between them is math
                    while '<EQ>' in line and '</EQ>' in line:
                        line = re.sub(r'<EQ>\s*(.*?)\s*</EQ>', r' \\[ \1 \\] ', line, count=1)
                    return line
                statement = '\n'.join(_convert_eq(l) for l in statement.split('\n'))

                # Remove any leftover raw $$ markers introduced earlier in existing content
                # (If present from original statement rather than our replacements.)
                # Convert balanced pairs $$ X $$ to \[ X \]
                statement = re.sub(r'\$\$\s*(.*?)\s*\$\$', r' \\[ \1 \\] ', statement)
                # Remove stray unmatched $$ at line ends
                statement = re.sub(r'\$\$\s*$', '', statement)

                # Final whitespace normalization
                statement = '\n'.join(l.strip() for l in statement.split('\n'))
                statement = re.sub(r'\n{3,}', '\n\n', statement).strip()
                
                # Look for associated proof
                proof_content = self._find_associated_proof(content, match.end())
                
                theorem = TheoremStatement(
                    statement_type=content_type,
                    number=None,  # TODO: Extract numbering
                    title=title,
                    statement=statement,
                    proof=proof_content
                )
                
                theorems.append(theorem)
        
        return theorems
    
    def _find_associated_proof(self, content: str, theorem_end_pos: int) -> Optional[str]:
        """Find proof that follows a theorem statement."""
        # Look for proof environment starting near the theorem end
        remaining_content = content[theorem_end_pos:]
        
        for proof_env in self.PROOF_ENVIRONMENTS:
            proof_pattern = rf'\\begin\{{{proof_env}\}}(.*?)\\end\{{{proof_env}\}}'
            
            match = re.search(proof_pattern, remaining_content, re.DOTALL | re.IGNORECASE)
            if match:
                # Check if proof starts reasonably close to theorem
                proof_start = match.start()
                intervening_content = remaining_content[:proof_start].strip()
                
                # If there's minimal content between theorem and proof, assume they're associated
                if len(intervening_content) < 200:  # Arbitrary threshold
                    return match.group(1).strip()
        
        return None
    
    def _extract_references(self, content: str) -> Optional[str]:
        """Extract the references/bibliography section."""
        # Look for bibliography environments or sections
        ref_patterns = [
            r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
            r'\\section\*?\{[Rr]eferences\}(.*?)(?=\\section|\Z)',
            r'\\bibliography\{[^}]+\}',
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def remove_proofs(self, content: str) -> str:
        """Remove all proofs from LaTeX content while preserving theorem statements."""
        modified_content = content
        
        # Remove proof environments
        for proof_env in self.PROOF_ENVIRONMENTS:
            pattern = rf'\\begin\{{{proof_env}\}}.*?\\end\{{{proof_env}\}}'
            modified_content = re.sub(pattern, '', modified_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up extra whitespace
        modified_content = re.sub(r'\n\s*\n\s*\n', '\n\n', modified_content)
        
        return modified_content
    
    def get_theorem_statements_only(self, theorems: List[TheoremStatement]) -> List[TheoremStatement]:
        """Return theorems with proofs removed."""
        return [
            TheoremStatement(
                statement_type=thm.statement_type,
                number=thm.number,
                title=thm.title,
                statement=thm.statement,
                proof=None,  # Remove proof
                line_start=thm.line_start,
                line_end=thm.line_end
            )
            for thm in theorems
        ]


def main():
    """Test the LaTeX parser with sample content."""
    sample_latex = r"""
    \documentclass{article}
    
    \begin{abstract}
    This paper proves important results about prime numbers and their distribution.
    We show several new theorems using innovative techniques.
    \end{abstract}
    
    \section{Introduction}
    Prime numbers have fascinated mathematicians for centuries. In this work, we 
    investigate their properties using novel methods from algebraic geometry.
    
    \section{Main Results}
    
    \begin{theorem}[Prime Distribution]
    There are infinitely many primes of the form $4n+1$.
    \end{theorem}
    
    \begin{proof}
    Assume for contradiction that there are only finitely many such primes.
    Let $p_1, p_2, \ldots, p_k$ be all such primes...
    \end{proof}
    
    \begin{lemma}
    Every prime $p > 2$ is odd.
    \end{lemma}
    
    \begin{proof}
    This follows from the definition of primality.
    \end{proof}
    
    \section{References}
    [1] Hardy, G.H. and Wright, E.M., An Introduction to the Theory of Numbers.
    """
    
    parser = LatexParser()
    parsed = parser.parse(sample_latex)
    
    print("=== ABSTRACT ===")
    print(parsed['abstract'])
    
    print("\n=== INTRODUCTION ===")
    print(parsed['introduction'])
    
    print("\n=== THEOREMS ===")
    for i, thm in enumerate(parsed['theorems'], 1):
        print(f"\n{i}. {thm.statement_type.value.upper()}")
        if thm.title:
            print(f"   Title: {thm.title}")
        print(f"   Statement: {thm.statement}")
        if thm.proof:
            print(f"   Has proof: Yes ({len(thm.proof)} chars)")
        else:
            print("   Has proof: No")
    
    print("\n=== CONTENT WITHOUT PROOFS ===")
    no_proofs = parser.remove_proofs(sample_latex)
    print(no_proofs[:500] + "..." if len(no_proofs) > 500 else no_proofs)


if __name__ == "__main__":
    main()