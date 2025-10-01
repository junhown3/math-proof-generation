"""
arXiv Paper Fetcher for Mathematical Research
Downloads LaTeX source files from arXiv for specific math subjects after a given date.
"""

import requests
import urllib.parse
try:
    import feedparser  # type: ignore
except ImportError as e:  # Provide clearer guidance if dependency missing
    raise ImportError(
        "Missing optional dependency 'feedparser'. Install it via 'pip install feedparser' or add it to requirements.txt. "
        "If you only want to generate proofs from already-downloaded/parsed papers, you can temporarily stub out arxiv fetching."
    ) from e
import tarfile
import io
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ArxivPaper:
    """Represents an arXiv paper with metadata and LaTeX content."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    subjects: List[str]
    published_date: datetime
    latex_content: Optional[str] = None
    main_tex_file: Optional[str] = None


class ArxivFetcher:
    """Fetches mathematical papers from arXiv with LaTeX source."""
    
    # arXiv subject categories for mathematics
    MATH_SUBJECTS = {
        'math.NT': 'Number Theory',
        'math.RT': 'Representation Theory', 
        'math.AG': 'Algebraic Geometry'
    }
    
    def __init__(self, base_url: str = "http://export.arxiv.org/api/query"):
        self.base_url = base_url
        
    def search_papers(self, subjects: List[str], start_date: datetime, 
                     end_date: Optional[datetime] = None, max_results: int = 10) -> List[ArxivPaper]:
        """
        Search for papers in specified subjects after start_date.
        
        Args:
            subjects: List of arXiv subject codes (e.g., ['math.NT', 'math.AG'])
            start_date: Only papers submitted after this date
            end_date: Only papers submitted before this date (optional)
            max_results: Maximum number of papers to return
            
        Returns:
            List of ArxivPaper objects with metadata
        """
        if not subjects:
            subjects = list(self.MATH_SUBJECTS.keys())
            
        # Build search query for arXiv API
        subject_query = " OR ".join([f"cat:{subj}" for subj in subjects])
        query = f"({subject_query})"
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        
        # Parse the Atom feed
        feed = feedparser.parse(response.content)
        papers = []
        
        for entry in feed.entries:
            # Extract arXiv ID
            arxiv_id = entry.id.split('/')[-1]
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.split('v')[0]  # Remove version number
                
            # Parse publication date
            published_date = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
            
            # Filter by date
            if published_date < start_date:
                continue
            if end_date and published_date > end_date:
                continue
                
            # Extract subjects/categories
            subjects_list = []
            if hasattr(entry, 'tags'):
                subjects_list = [tag['term'] for tag in entry.tags]
            
            # Extract authors
            authors = []
            if hasattr(entry, 'authors'):
                authors = [author['name'] for author in entry.authors]
            elif hasattr(entry, 'author'):
                authors = [entry.author]
                
            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                title=entry.title.replace('\n', ' ').strip(),
                authors=authors,
                abstract=entry.summary.replace('\n', ' ').strip(),
                subjects=subjects_list,
                published_date=published_date
            )
            
            papers.append(paper)
            
        return papers
    
    def download_latex_source(self, paper: ArxivPaper) -> bool:
        """
        Download and extract LaTeX source for a paper.
        
        Args:
            paper: ArxivPaper object to download source for
            
        Returns:
            True if successful, False otherwise
        """
        source_url = f"https://arxiv.org/src/{paper.arxiv_id}"
        
        try:
            response = requests.get(source_url)
            response.raise_for_status()
            
            # Extract tar.gz content
            tar_data = io.BytesIO(response.content)
            
            with tarfile.open(fileobj=tar_data, mode='r:gz') as tar:
                # Find the main .tex file
                tex_files = [name for name in tar.getnames() if name.endswith('.tex')]
                
                if not tex_files:
                    print(f"No .tex files found for paper {paper.arxiv_id}")
                    return False
                
                # Try to identify the main tex file
                main_tex = self._identify_main_tex_file(tex_files, tar)
                
                if main_tex:
                    # Extract the main tex file content
                    tex_file = tar.extractfile(main_tex)
                    if tex_file:
                        paper.latex_content = tex_file.read().decode('utf-8', errors='ignore')
                        paper.main_tex_file = main_tex
                        return True
                        
        except Exception as e:
            print(f"Error downloading source for {paper.arxiv_id}: {e}")
            return False
            
        return False
    
    def _identify_main_tex_file(self, tex_files: List[str], tar: tarfile.TarFile) -> Optional[str]:
        """
        Try to identify the main LaTeX file among multiple .tex files.
        
        Args:
            tex_files: List of .tex filenames
            tar: TarFile object to read from
            
        Returns:
            Name of the main .tex file, or None if not found
        """
        if len(tex_files) == 1:
            return tex_files[0]
        
        # Look for common main file names
        main_candidates = ['main.tex', 'paper.tex', 'manuscript.tex']
        for candidate in main_candidates:
            if candidate in tex_files:
                return candidate
        
        # Look for files with \documentclass
        for tex_file in tex_files:
            try:
                file_obj = tar.extractfile(tex_file)
                if file_obj:
                    content = file_obj.read().decode('utf-8', errors='ignore')
                    if r'\documentclass' in content:
                        return tex_file
            except:
                continue
        
        # Fallback: return the first .tex file
        return tex_files[0] if tex_files else None


def main():
    """Example usage of the ArxivFetcher."""
    fetcher = ArxivFetcher()
    
    # Search for recent papers in number theory and algebraic geometry
    # Using August 2024 as cutoff (approximate GPT-4o training cutoff)
    start_date = datetime(2024, 8, 1)
    subjects = ['math.NT', 'math.AG']
    
    print("Searching for recent papers...")
    papers = fetcher.search_papers(subjects, start_date, max_results=5)
    
    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   arXiv ID: {paper.arxiv_id}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   Date: {paper.published_date.strftime('%Y-%m-%d')}")
        print(f"   Subjects: {', '.join(paper.subjects)}")
        
        # Try to download LaTeX source
        print("   Downloading LaTeX source...", end=" ")
        if fetcher.download_latex_source(paper):
            print(f"✓ Success ({len(paper.latex_content)} chars)")
        else:
            print("✗ Failed")


if __name__ == "__main__":
    main()