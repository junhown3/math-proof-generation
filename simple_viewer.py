"""
Simple Data Viewer - Non-interactive version to examine the saved data
"""

import json
import os


def show_all_data():
    """Show all saved data in a simple format."""
    
    print("="*60)
    print("MATH PAPER DATA VIEWER")
    print("="*60)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("No data directory found. Run data_manager.py first.")
        return
    
    # List papers
    papers_dir = "data/papers"
    if not os.path.exists(papers_dir):
        print("No papers directory found.")
        return
    
    paper_files = [f for f in os.listdir(papers_dir) if f.endswith('.json')]
    
    if not paper_files:
        print("No papers found.")
        return
    
    print(f"\nFound {len(paper_files)} saved papers:")
    
    for i, filename in enumerate(paper_files, 1):
        filepath = os.path.join(papers_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        print(f"\n{i}. {paper_data['title']}")
        print(f"   arXiv ID: {paper_data['arxiv_id']}")
        print(f"   Authors: {', '.join(paper_data['authors'])}")
        print(f"   Subjects: {', '.join(paper_data['subjects'])}")
        print(f"   Published: {paper_data['published_date'][:10]}")
        print(f"   LaTeX size: {len(paper_data['latex_content']):,} characters")
        
        # Show abstract
        print(f"   Abstract: {paper_data['abstract'][:100]}...")
        
        # Check for parsed data
        arxiv_id = paper_data['arxiv_id']
        parsed_file = f"data/parsed/{arxiv_id.replace('/', '_')}_parsed.json"
        
        if os.path.exists(parsed_file):
            with open(parsed_file, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)
            
            theorems = parsed_data.get('theorems', [])
            sections = parsed_data.get('sections', [])
            
            print(f"   Parsed: ✓ {len(theorems)} theorems, {len(sections)} sections")
            
            # Show theorem types
            theorem_types = {}
            theorems_with_proofs = 0
            
            for thm in theorems:
                thm_type = thm.get('statement_type', 'unknown')
                theorem_types[thm_type] = theorem_types.get(thm_type, 0) + 1
                if thm.get('proof'):
                    theorems_with_proofs += 1
            
            type_summary = ', '.join([f"{count} {thm_type}" for thm_type, count in theorem_types.items()])
            print(f"   Theorem breakdown: {type_summary}")
            print(f"   Theorems with proofs: {theorems_with_proofs}/{len(theorems)}")
            
        else:
            print(f"   Parsed: ✗ No parsed data found")


def show_first_theorem():
    """Show the first theorem from the first paper as an example."""
    
    papers_dir = "data/papers"
    paper_files = [f for f in os.listdir(papers_dir) if f.endswith('.json')]
    
    if not paper_files:
        return
    
    # Get first paper
    with open(os.path.join(papers_dir, paper_files[0]), 'r', encoding='utf-8') as f:
        paper_data = json.load(f)
    
    arxiv_id = paper_data['arxiv_id']
    parsed_file = f"data/parsed/{arxiv_id.replace('/', '_')}_parsed.json"
    
    if not os.path.exists(parsed_file):
        return
    
    with open(parsed_file, 'r', encoding='utf-8') as f:
        parsed_data = json.load(f)
    
    theorems = parsed_data.get('theorems', [])
    
    if not theorems:
        return
    
    print(f"\n{'='*60}")
    print(f"EXAMPLE: FIRST THEOREM FROM {arxiv_id}")
    print(f"{'='*60}")
    
    first_theorem = theorems[0]
    
    print(f"Type: {first_theorem.get('statement_type', 'unknown').upper()}")
    
    if first_theorem.get('title'):
        print(f"Title: {first_theorem['title']}")
    
    print(f"\nStatement:")
    statement = first_theorem.get('statement', '')
    print(statement[:500] + "..." if len(statement) > 500 else statement)
    
    if first_theorem.get('proof'):
        proof = first_theorem['proof']
        print(f"\nProof ({len(proof):,} characters):")
        print(proof[:300] + "..." if len(proof) > 300 else proof)
    else:
        print("\nProof: Not found")


def show_latex_sample():
    """Show a sample of the raw LaTeX content."""
    
    papers_dir = "data/papers"
    paper_files = [f for f in os.listdir(papers_dir) if f.endswith('.json')]
    
    if not paper_files:
        return
    
    # Get first paper
    with open(os.path.join(papers_dir, paper_files[0]), 'r', encoding='utf-8') as f:
        paper_data = json.load(f)
    
    latex_content = paper_data.get('latex_content', '')
    
    if not latex_content:
        return
    
    print(f"\n{'='*60}")
    print(f"RAW LATEX SAMPLE FROM {paper_data['arxiv_id']}")
    print(f"{'='*60}")
    
    print(f"Total length: {len(latex_content):,} characters")
    print(f"Main file: {paper_data.get('main_tex_file', 'unknown')}")
    
    print(f"\nFirst 800 characters:")
    print("-" * 50)
    print(latex_content[:800])
    print("-" * 50)


if __name__ == "__main__":
    show_all_data()
    show_first_theorem()
    show_latex_sample()