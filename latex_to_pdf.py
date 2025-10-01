"""
LaTeX to PDF Converter
Converts processed mathematical papers to PDF using TeXworks/pdflatex.
"""

import os
import re
import subprocess
import tempfile
import shutil
from typing import Optional, Tuple, Dict
from data_manager import DataManager
from latex_parser import LatexParser


class LatexToPdfConverter:
    """Converts processed LaTeX content to readable PDFs."""
    
    def __init__(self, texworks_path: str = None):
        """
        Initialize converter.
        
        Args:
            texworks_path: Path to TeXworks or pdflatex executable
        """
        self.texworks_path = texworks_path or self._find_pdflatex()
        self.data_manager = DataManager()
        self.parser = LatexParser()
    
    def _find_pdflatex(self) -> Optional[str]:
        """Try to find pdflatex executable."""
        # Common locations for pdflatex on Windows
        common_paths = [
            r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
            r"C:\texlive\2023\bin\win32\pdflatex.exe",
            r"C:\texlive\2024\bin\windows\pdflatex.exe",
            "pdflatex",  # If in PATH
        ]
        
        for path in common_paths:
            if os.path.exists(path) or path == "pdflatex":
                try:
                    # Test if pdflatex works
                    result = subprocess.run([path, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        print(f"Found pdflatex at: {path}")
                        return path
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
        
        return None
    
    def create_pdf_from_paper(self, arxiv_id: str, remove_proofs: bool = True, 
                             output_dir: str = "pdfs") -> Optional[str]:
        """
        Create PDF from a saved paper.
        
        Args:
            arxiv_id: arXiv ID of the paper
            remove_proofs: Whether to remove proofs before PDF generation
            output_dir: Directory to save PDFs
            
        Returns:
            Path to generated PDF or None if failed
        """
        if not self.texworks_path:
            print("Error: pdflatex not found. Please install TeXworks/MiKTeX or specify path.")
            return None
        
        # Load the paper
        paper = self.data_manager.load_paper(arxiv_id)
        if not paper:
            print(f"Paper {arxiv_id} not found.")
            return None
        
        # Process the LaTeX content
        latex_content = paper.latex_content
        
        if remove_proofs:
            latex_content = self.parser.remove_proofs(latex_content)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate PDF
        pdf_path = self._compile_latex_to_pdf(
            latex_content, arxiv_id, paper.title, output_dir, remove_proofs
        )
        
        return pdf_path
    
    def _compile_latex_to_pdf(self, latex_content: str, arxiv_id: str, title: str,
                             output_dir: str, proofs_removed: bool) -> Optional[str]:
        """Compile LaTeX content to PDF."""
        
        # Clean and prepare LaTeX content
        processed_content = self._prepare_latex_content(latex_content, title, proofs_removed)
        
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write LaTeX file
            tex_filename = f"{arxiv_id.replace('/', '_')}.tex"
            tex_path = os.path.join(temp_dir, tex_filename)
            
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            # Compile to PDF (run twice for proper references)
            for attempt in range(2):
                try:
                    result = subprocess.run(
                        [self.texworks_path, "-interaction=nonstopmode", "-output-directory", temp_dir, tex_path],
                        capture_output=True, text=True, timeout=60, cwd=temp_dir
                    )
                    
                    if attempt == 0:  # First run - show any critical errors
                        if result.returncode != 0 and "Emergency stop" in result.stdout:
                            print(f"LaTeX compilation failed on first attempt:")
                            print(result.stdout[-1000:])  # Last 1000 chars of output
                            return None
                
                except subprocess.TimeoutExpired:
                    print("LaTeX compilation timed out.")
                    return None
            
            # Check if PDF was created
            pdf_temp_path = os.path.join(temp_dir, f"{arxiv_id.replace('/', '_')}.pdf")
            
            if not os.path.exists(pdf_temp_path):
                print("PDF compilation failed. LaTeX output:")
                print(result.stdout[-1000:])
                return None
            
            # Copy PDF to output directory
            suffix = "_no_proofs" if proofs_removed else "_full"
            pdf_filename = f"{arxiv_id.replace('/', '_')}{suffix}.pdf"
            pdf_output_path = os.path.join(output_dir, pdf_filename)
            
            shutil.copy2(pdf_temp_path, pdf_output_path)
            
            print(f"✓ PDF created: {pdf_output_path}")
            return pdf_output_path
    
    def _prepare_latex_content(self, latex_content: str, title: str, proofs_removed: bool) -> str:
        """Prepare LaTeX content for compilation."""
        
        # Extract existing preamble if present
        preamble = self._extract_preamble(latex_content)
        body = self._extract_body(latex_content)
        
        # If no proper document structure found, create a wrapper
        if not preamble or not body:
            return self._create_wrapper_document(latex_content, title, proofs_removed)
        
        # Use existing structure but ensure it compiles
        content = preamble + "\n\n" + body
        
        # Add note about proof removal
        if proofs_removed:
            content = content.replace(
                r"\begin{document}",
                r"\begin{document}" + "\n\n" + 
                r"\noindent\textbf{Note:} Proofs have been removed from this version for analysis purposes." + 
                "\n\n"
            )
        
        return content
    
    def _extract_preamble(self, latex_content: str) -> Optional[str]:
        """Extract preamble (everything before \begin{document})."""
        match = re.search(r'(.*?)\\begin\{document\}', latex_content, re.DOTALL)
        
        if match:
            preamble = match.group(1).strip()
            # Ensure it has documentclass
            if r'\documentclass' in preamble:
                return preamble + "\n\\begin{document}"
        
        return None
    
    def _extract_body(self, latex_content: str) -> Optional[str]:
        """Extract body (everything between \begin{document} and \end{document})."""
        match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
        
        if match:
            return match.group(1).strip() + "\n\\end{document}"
        
        return None
    
    def _create_wrapper_document(self, content: str, title: str, proofs_removed: bool) -> str:
        """Create a complete LaTeX document wrapper."""
        
        # Remove any existing documentclass/begin{document}/end{document}
        content = re.sub(r'\\documentclass.*?\n', '', content)
        content = re.sub(r'\\begin\{document\}', '', content)
        content = re.sub(r'\\end\{document\}', '', content)
        
        # Standard mathematical preamble
        preamble = r'''\documentclass[11pt]{amsart}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

% Common theorem environments
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

% Common math commands
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}

\title{''' + title.replace('_', r'\_') + r'''}
\author{arXiv Paper - Processed Version}

\begin{document}

\maketitle'''

        if proofs_removed:
            preamble += r'''

\begin{center}
\textbf{\large Note: Proofs have been removed from this version for analysis purposes.}
\end{center}'''

        # Combine everything
        full_document = preamble + "\n\n" + content + "\n\n\\end{document}"
        
        return full_document


def main():
    """Test the LaTeX to PDF converter."""
    converter = LatexToPdfConverter()
    
    if not converter.texworks_path:
        print("Could not find pdflatex. Please install TeXworks/MiKTeX.")
        return
    
    # List available papers
    data_manager = DataManager()
    papers = data_manager.list_saved_papers()
    
    if not papers:
        print("No papers found. Run data_manager.py first.")
        return
    
    print("Available papers:")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title'][:60]}... (ID: {paper['arxiv_id']})")
    
    # Convert first paper as test
    first_paper = papers[0]
    arxiv_id = first_paper['arxiv_id']
    
    print(f"\nConverting paper {arxiv_id} to PDF...")
    
    # Create both versions: with and without proofs
    print("\n1. Creating version WITH proofs...")
    pdf_path_full = converter.create_pdf_from_paper(arxiv_id, remove_proofs=False)
    
    print("\n2. Creating version WITHOUT proofs...")
    pdf_path_no_proofs = converter.create_pdf_from_paper(arxiv_id, remove_proofs=True)
    
    print(f"\nResults:")
    if pdf_path_full:
        print(f"✓ Full version: {pdf_path_full}")
    else:
        print("✗ Full version failed")
    
    if pdf_path_no_proofs:
        print(f"✓ No proofs version: {pdf_path_no_proofs}")
    else:
        print("✗ No proofs version failed")


if __name__ == "__main__":
    main()