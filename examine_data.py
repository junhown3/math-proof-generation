"""
Data Examination Tool
Interactive tool to examine saved paper data.
"""

import json
import os
from data_manager import DataManager


def main():
    """Interactive data examination tool."""
    data_manager = DataManager()
    
    while True:
        print(f"\n{'='*60}")
        print("MATH PAPER DATA EXAMINATION TOOL")
        print(f"{'='*60}")
        
        # List available papers
        papers = data_manager.list_saved_papers()
        
        if not papers:
            print("No papers found. Run data_manager.py first to fetch papers.")
            return
        
        print(f"\nAvailable papers ({len(papers)}):")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {paper['title'][:50]}...")
            print(f"   arXiv: {paper['arxiv_id']} | Size: {paper['latex_size']:,} chars")
        
        print(f"\nOptions:")
        print("1-{}: Examine paper in detail".format(len(papers)))
        print("t: Show theorem list for a paper")
        print("s: Show sections for a paper")  
        print("r: Show raw LaTeX excerpt")
        print("p: Show parsed content stats")
        print("q: Quit")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(papers):
            # Examine paper in detail
            paper = papers[int(choice) - 1]
            data_manager.examine_paper(paper['arxiv_id'])
            input("\nPress Enter to continue...")
        elif choice == 't':
            # Show theorems
            paper_num = input(f"Enter paper number (1-{len(papers)}): ")
            if paper_num.isdigit() and 1 <= int(paper_num) <= len(papers):
                show_theorems(papers[int(paper_num) - 1]['arxiv_id'])
            input("\nPress Enter to continue...")
        elif choice == 's':
            # Show sections
            paper_num = input(f"Enter paper number (1-{len(papers)}): ")
            if paper_num.isdigit() and 1 <= int(paper_num) <= len(papers):
                show_sections(papers[int(paper_num) - 1]['arxiv_id'])
            input("\nPress Enter to continue...")
        elif choice == 'r':
            # Show raw LaTeX
            paper_num = input(f"Enter paper number (1-{len(papers)}): ")
            if paper_num.isdigit() and 1 <= int(paper_num) <= len(papers):
                show_raw_latex(papers[int(paper_num) - 1]['arxiv_id'])
            input("\nPress Enter to continue...")
        elif choice == 'p':
            # Show parsed content stats
            paper_num = input(f"Enter paper number (1-{len(papers)}): ")
            if paper_num.isdigit() and 1 <= int(paper_num) <= len(papers):
                show_parsed_stats(papers[int(paper_num) - 1]['arxiv_id'])
            input("\nPress Enter to continue...")


def show_theorems(arxiv_id: str):
    """Show all theorems in a paper."""
    parsed_file = f"data/parsed/{arxiv_id.replace('/', '_')}_parsed.json"
    
    if not os.path.exists(parsed_file):
        print("Parsed content not found.")
        return
    
    with open(parsed_file, 'r', encoding='utf-8') as f:
        parsed = json.load(f)
    
    theorems = parsed.get('theorems', [])
    
    print(f"\n{'='*50}")
    print(f"THEOREMS IN {arxiv_id}")
    print(f"{'='*50}")
    
    if not theorems:
        print("No theorems found.")
        return
    
    for i, thm in enumerate(theorems, 1):
        print(f"\n{i}. {thm['statement_type'].upper()}")
        if thm.get('title'):
            print(f"   Title: {thm['title']}")
        
        statement = thm['statement']
        if len(statement) > 200:
            statement = statement[:200] + "..."
        print(f"   Statement: {statement}")
        
        if thm.get('proof'):
            proof_len = len(thm['proof'])
            print(f"   Proof: {proof_len:,} characters")
            
            # Show first few lines of proof
            proof_lines = thm['proof'].split('\n')[:3]
            proof_preview = ' '.join(proof_lines)
            if len(proof_preview) > 150:
                proof_preview = proof_preview[:150] + "..."
            print(f"   Preview: {proof_preview}")
        else:
            print("   Proof: Not found")


def show_sections(arxiv_id: str):
    """Show all sections in a paper."""
    parsed_file = f"data/parsed/{arxiv_id.replace('/', '_')}_parsed.json"
    
    if not os.path.exists(parsed_file):
        print("Parsed content not found.")
        return
    
    with open(parsed_file, 'r', encoding='utf-8') as f:
        parsed = json.load(f)
    
    sections = parsed.get('sections', [])
    
    print(f"\n{'='*50}")
    print(f"SECTIONS IN {arxiv_id}")
    print(f"{'='*50}")
    
    if not sections:
        print("No sections found.")
        return
    
    for i, section in enumerate(sections, 1):
        print(f"\n{i}. {section['title']}")
        print(f"   Type: {section['content_type']}")
        print(f"   Length: {len(section['content']):,} characters")
        
        # Show first few lines
        content_lines = section['content'].split('\n')[:3]
        preview = ' '.join(content_lines)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        print(f"   Preview: {preview}")


def show_raw_latex(arxiv_id: str):
    """Show raw LaTeX content excerpt."""
    data_manager = DataManager()
    paper = data_manager.load_paper(arxiv_id)
    
    if not paper:
        print("Paper not found.")
        return
    
    print(f"\n{'='*50}")
    print(f"RAW LATEX EXCERPT: {arxiv_id}")
    print(f"{'='*50}")
    
    latex = paper.latex_content
    
    print(f"Total length: {len(latex):,} characters")
    print(f"Main tex file: {paper.main_tex_file}")
    
    print(f"\nFirst 1000 characters:")
    print("-" * 40)
    print(latex[:1000])
    print("-" * 40)
    
    print(f"\nLast 500 characters:")
    print("-" * 40)
    print(latex[-500:])
    print("-" * 40)


def show_parsed_stats(arxiv_id: str):
    """Show parsed content statistics."""
    parsed_file = f"data/parsed/{arxiv_id.replace('/', '_')}_parsed.json"
    
    if not os.path.exists(parsed_file):
        print("Parsed content not found.")
        return
    
    with open(parsed_file, 'r', encoding='utf-8') as f:
        parsed = json.load(f)
    
    print(f"\n{'='*50}")
    print(f"PARSED CONTENT STATISTICS: {arxiv_id}")
    print(f"{'='*50}")
    
    abstract = parsed.get('abstract', '')
    intro = parsed.get('introduction', '')
    sections = parsed.get('sections', [])
    theorems = parsed.get('theorems', [])
    references = parsed.get('references', '')
    
    print(f"Abstract: {'✓' if abstract else '✗'} ({len(abstract):,} chars)")
    print(f"Introduction: {'✓' if intro else '✗'} ({len(intro):,} chars)")
    print(f"Sections: {len(sections)}")
    
    total_section_content = sum(len(s.get('content', '')) for s in sections)
    print(f"  Total section content: {total_section_content:,} chars")
    
    print(f"Theorems: {len(theorems)}")
    
    theorem_types = {}
    total_proofs = 0
    total_proof_length = 0
    
    for thm in theorems:
        thm_type = thm.get('statement_type', 'unknown')
        theorem_types[thm_type] = theorem_types.get(thm_type, 0) + 1
        
        if thm.get('proof'):
            total_proofs += 1
            total_proof_length += len(thm['proof'])
    
    for thm_type, count in theorem_types.items():
        print(f"  {thm_type}: {count}")
    
    print(f"  With proofs: {total_proofs}/{len(theorems)}")
    print(f"  Total proof content: {total_proof_length:,} chars")
    
    print(f"References: {'✓' if references else '✗'} ({len(references):,} chars)")


if __name__ == "__main__":
    main()