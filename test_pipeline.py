"""
Test the complete pipeline: fetch a real arXiv paper and parse it.
"""

from arxiv_fetcher import ArxivFetcher
from latex_parser import LatexParser
from datetime import datetime


def test_complete_pipeline():
    """Test fetching and parsing a real arXiv paper."""
    
    # Initialize components
    fetcher = ArxivFetcher()
    parser = LatexParser()
    
    # Fetch a recent paper
    print("Fetching recent mathematical papers...")
    start_date = datetime(2024, 8, 1)
    papers = fetcher.search_papers(['math.NT'], start_date, max_results=3)
    
    if not papers:
        print("No papers found!")
        return
    
    # Process the first paper
    paper = papers[0]
    print(f"\nProcessing paper: {paper.title}")
    print(f"arXiv ID: {paper.arxiv_id}")
    
    # Download LaTeX source
    if fetcher.download_latex_source(paper):
        print(f"✓ Downloaded LaTeX source ({len(paper.latex_content)} characters)")
        
        # Parse the content
        print("\nParsing LaTeX content...")
        parsed = parser.parse(paper.latex_content)
        
        # Display results
        print("\n" + "="*50)
        print("PARSING RESULTS")
        print("="*50)
        
        if parsed['abstract']:
            print("\n--- ABSTRACT ---")
            print(parsed['abstract'][:300] + "..." if len(parsed['abstract']) > 300 else parsed['abstract'])
        
        if parsed['introduction']:
            print("\n--- INTRODUCTION ---")
            print(parsed['introduction'][:400] + "..." if len(parsed['introduction']) > 400 else parsed['introduction'])
        
        print(f"\n--- SECTIONS ---")
        print(f"Found {len(parsed['sections'])} sections")
        for section in parsed['sections'][:3]:  # Show first 3 sections
            print(f"  • {section.content_type.value}: {section.title}")
        
        print(f"\n--- THEOREMS ---")
        theorems = parsed['theorems']
        print(f"Found {len(theorems)} theorem-like statements")
        
        for i, thm in enumerate(theorems[:5], 1):  # Show first 5 theorems
            print(f"\n{i}. {thm.statement_type.value.upper()}")
            if thm.title:
                print(f"   Title: {thm.title}")
            print(f"   Statement: {thm.statement[:200]}...")
            print(f"   Has proof: {'Yes' if thm.proof else 'No'}")
            if thm.proof:
                print(f"   Proof length: {len(thm.proof)} characters")
        
        # Test proof removal
        print(f"\n--- PROOF REMOVAL ---")
        original_length = len(paper.latex_content)
        no_proofs_content = parser.remove_proofs(paper.latex_content)
        new_length = len(no_proofs_content)
        
        print(f"Original content: {original_length} characters")
        print(f"After proof removal: {new_length} characters")
        print(f"Reduction: {original_length - new_length} characters ({((original_length - new_length) / original_length * 100):.1f}%)")
        
    else:
        print("✗ Failed to download LaTeX source")


if __name__ == "__main__":
    test_complete_pipeline()