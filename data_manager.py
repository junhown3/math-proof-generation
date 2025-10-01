"""
Data Manager for Math Problem Solver
Handles saving, loading, and examining processed paper data.
"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from arxiv_fetcher import ArxivPaper
from latex_parser import LatexParser, TheoremStatement, ContentType
from context_preparator import ContextPreparator, FormattedContext


class DataManager:
    """Manages persistence and examination of processed paper data."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data manager.
        
        Args:
            data_dir: Directory to store processed data
        """
        self.data_dir = data_dir
        self.papers_dir = os.path.join(data_dir, "papers")
        self.parsed_dir = os.path.join(data_dir, "parsed")
        self.contexts_dir = os.path.join(data_dir, "contexts")
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.papers_dir, self.parsed_dir, self.contexts_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_paper(self, paper: ArxivPaper) -> str:
        """
        Save a paper with its LaTeX content.
        
        Args:
            paper: ArxivPaper object to save
            
        Returns:
            Path to saved file
        """
        filename = f"{paper.arxiv_id.replace('/', '_')}.json"
        filepath = os.path.join(self.papers_dir, filename)
        
        # Convert paper to dictionary
        paper_data = {
            'arxiv_id': paper.arxiv_id,
            'title': paper.title,
            'authors': paper.authors,
            'abstract': paper.abstract,
            'subjects': paper.subjects,
            'published_date': paper.published_date.isoformat(),
            'latex_content': paper.latex_content,
            'main_tex_file': paper.main_tex_file,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(paper_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved paper {paper.arxiv_id} to {filepath}")
        return filepath
    
    def save_parsed_content(self, paper_id: str, parsed_content: Dict) -> str:
        """
        Save parsed content from a paper.
        
        Args:
            paper_id: arXiv ID of the paper
            parsed_content: Dictionary with parsed content
            
        Returns:
            Path to saved file
        """
        filename = f"{paper_id.replace('/', '_')}_parsed.json"
        filepath = os.path.join(self.parsed_dir, filename)
        
        # Convert theorem objects to dictionaries
        serializable_content = self._make_serializable(parsed_content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_content, f, indent=2, ensure_ascii=False)
        
        print(f"Saved parsed content for {paper_id} to {filepath}")
        return filepath
    
    def save_context(self, paper_id: str, theorem_index: int, context: FormattedContext) -> str:
        """
        Save formatted context for a specific theorem.
        
        Args:
            paper_id: arXiv ID of the paper
            theorem_index: Index of the theorem
            context: FormattedContext object
            
        Returns:
            Path to saved file
        """
        filename = f"{paper_id.replace('/', '_')}_theorem_{theorem_index}_context.json"
        filepath = os.path.join(self.contexts_dir, filename)
        
        context_data = {
            'paper_metadata': context.paper_metadata,
            'paper_context': context.paper_context,
            'theorem_to_prove': self._theorem_to_dict(context.theorem_to_prove),
            'available_theorems': [self._theorem_to_dict(thm) for thm in context.available_theorems],
            'strategy_summary': context.strategy_summary,
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved context for {paper_id} theorem {theorem_index} to {filepath}")
        return filepath
    
    def load_paper(self, paper_id: str) -> Optional[ArxivPaper]:
        """Load a saved paper."""
        filename = f"{paper_id.replace('/', '_')}.json"
        filepath = os.path.join(self.papers_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paper = ArxivPaper(
            arxiv_id=data['arxiv_id'],
            title=data['title'],
            authors=data['authors'],
            abstract=data['abstract'],
            subjects=data['subjects'],
            published_date=datetime.fromisoformat(data['published_date']),
            latex_content=data['latex_content'],
            main_tex_file=data['main_tex_file']
        )
        
        return paper

    def load_parsed_only_as_paper(self, paper_id: str) -> Optional[ArxivPaper]:
        """Construct a minimal ArxivPaper from parsed JSON when original paper JSON not present.

        This enables an offline mode where only `data/parsed/<id>_parsed.json` exists.
        We synthesize metadata placeholders where not available.
        """
        parsed_filename = f"{paper_id.replace('/', '_')}_parsed.json"
        parsed_path = os.path.join(self.parsed_dir, parsed_filename)
        if not os.path.exists(parsed_path):
            return None
        try:
            with open(parsed_path, 'r', encoding='utf-8') as f:
                parsed = json.load(f)
        except Exception:
            return None

        # Minimal synthetic paper: attempt to recover title/metadata if embedded (not currently stored in parsed)
        title = parsed.get('title') or f"Paper {paper_id} (offline)"
        authors = parsed.get('authors') or []
        subjects = parsed.get('subjects') or []
        from datetime import datetime
        published_date = datetime(1970,1,1)
        # We do NOT have original LaTeX; reconstruct a pseudo LaTeX body by concatenating sections and theorems.
        sections_txt = []
        for s in parsed.get('sections', []):
            sections_txt.append(f"\\section*{{{s.get('title','')}}}\n{s.get('content','')}")
        latex_content = '\n\n'.join(sections_txt)

        return ArxivPaper(
            arxiv_id=paper_id,
            title=title,
            authors=authors,
            abstract=parsed.get('abstract', ''),
            subjects=subjects,
            published_date=published_date,
            latex_content=latex_content,
            main_tex_file=None
        )
    
    def list_saved_papers(self) -> List[Dict]:
        """List all saved papers with basic info."""
        papers = []
        
        for filename in os.listdir(self.papers_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.papers_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                papers.append({
                    'arxiv_id': data['arxiv_id'],
                    'title': data['title'],
                    'authors': data['authors'],
                    'subjects': data['subjects'],
                    'published_date': data['published_date'],
                    'saved_at': data.get('saved_at', 'unknown'),
                    'latex_size': len(data.get('latex_content', '')) if data.get('latex_content') else 0
                })
        
        # Sort by publication date (newest first)
        papers.sort(key=lambda x: x['published_date'], reverse=True)
        return papers
    
    def examine_paper(self, paper_id: str) -> None:
        """Print detailed examination of a saved paper."""
        paper = self.load_paper(paper_id)
        if not paper:
            print(f"Paper {paper_id} not found")
            return
        
        print(f"\n{'='*60}")
        print(f"PAPER EXAMINATION: {paper.arxiv_id}")
        print(f"{'='*60}")
        
        print(f"\n📄 METADATA")
        print(f"Title: {paper.title}")
        print(f"Authors: {', '.join(paper.authors)}")
        print(f"Subjects: {', '.join(paper.subjects)}")
        print(f"Published: {paper.published_date.strftime('%Y-%m-%d')}")
        print(f"LaTeX size: {len(paper.latex_content):,} characters")
        print(f"Main file: {paper.main_tex_file}")
        
        print(f"\n📝 ABSTRACT")
        print(paper.abstract)
        
        # Parse the content to show structure
        parser = LatexParser()
        parsed = parser.parse(paper.latex_content)
        
        print(f"\n🏗️ PAPER STRUCTURE")
        if parsed['introduction']:
            print(f"✓ Introduction: {len(parsed['introduction']):,} chars")
        else:
            print("✗ No introduction found")
        
        print(f"✓ Sections: {len(parsed['sections'])}")
        for i, section in enumerate(parsed['sections'][:5], 1):
            print(f"  {i}. {section.title} ({len(section.content):,} chars)")
        if len(parsed['sections']) > 5:
            print(f"  ... and {len(parsed['sections']) - 5} more sections")
        
        print(f"✓ Theorems: {len(parsed['theorems'])}")
        for i, thm in enumerate(parsed['theorems'][:3], 1):
            title = f" - {thm.title}" if thm.title else ""
            proof_info = f" (with proof: {len(thm.proof):,} chars)" if thm.proof else " (no proof)"
            print(f"  {i}. {thm.statement_type.value.title()}{title}{proof_info}")
        if len(parsed['theorems']) > 3:
            print(f"  ... and {len(parsed['theorems']) - 3} more theorems")
        
        # Show proof removal statistics
        no_proofs = parser.remove_proofs(paper.latex_content)
        reduction = len(paper.latex_content) - len(no_proofs)
        reduction_pct = (reduction / len(paper.latex_content)) * 100
        
        print(f"\n✂️ PROOF REMOVAL")
        print(f"Original: {len(paper.latex_content):,} chars")
        print(f"After removal: {len(no_proofs):,} chars")
        print(f"Reduction: {reduction:,} chars ({reduction_pct:.1f}%)")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, TheoremStatement):
            return self._theorem_to_dict(obj)
        elif isinstance(obj, ContentType):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(dict(obj.__dict__))
        else:
            try:
                json.dumps(obj)  # Test if it's JSON serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)  # Convert to string as fallback
    
    def _theorem_to_dict(self, theorem: TheoremStatement) -> Dict:
        """Convert TheoremStatement to dictionary."""
        return {
            'statement_type': theorem.statement_type.value,
            'number': theorem.number,
            'title': theorem.title,
            'statement': theorem.statement,
            'proof': theorem.proof,
            'line_start': theorem.line_start,
            'line_end': theorem.line_end
        }


def main():
    """Test the data manager with real papers."""
    from arxiv_fetcher import ArxivFetcher
    
    # Initialize components
    fetcher = ArxivFetcher()
    data_manager = DataManager()
    parser = LatexParser()
    
    # Fetch and save some papers
    print("Fetching recent papers...")
    start_date = datetime(2024, 8, 1)
    papers = fetcher.search_papers(['math.NT'], start_date, max_results=2)
    
    saved_papers = []
    for paper in papers:
        if fetcher.download_latex_source(paper):
            # Save the paper
            data_manager.save_paper(paper)
            
            # Parse and save the parsed content
            parsed = parser.parse(paper.latex_content)
            data_manager.save_parsed_content(paper.arxiv_id, parsed)
            
            saved_papers.append(paper)
    
    # List all saved papers
    print(f"\n{'='*50}")
    print("SAVED PAPERS")
    print(f"{'='*50}")
    
    saved_list = data_manager.list_saved_papers()
    for i, paper_info in enumerate(saved_list, 1):
        print(f"\n{i}. {paper_info['title'][:60]}...")
        print(f"   arXiv: {paper_info['arxiv_id']}")
        print(f"   Authors: {', '.join(paper_info['authors'][:2])}{'...' if len(paper_info['authors']) > 2 else ''}")
        print(f"   Size: {paper_info['latex_size']:,} chars")
        print(f"   Published: {paper_info['published_date'][:10]}")
    
    # Examine the first paper in detail
    if saved_list:
        print(f"\n{'='*50}")
        print("DETAILED EXAMINATION")
        print(f"{'='*50}")
        first_paper = saved_list[0]
        data_manager.examine_paper(first_paper['arxiv_id'])


if __name__ == "__main__":
    main()