"""
Mathematical Proof Agent
Uses GPT-OSS on RunPod GPU or API fallback to generate mathematical proofs.
"""

import json
import os
import time
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import re
from data_manager import DataManager
from context_preparator import ContextPreparator, FormattedContext
from rl.structural_validators import run_structural_validators


@dataclass
class ProofResult:
    """Result of a proof generation attempt."""
    theorem_statement: str
    generated_proof: str
    model_used: str
    generation_time: float
    success: bool
    error_message: Optional[str] = None
    token_count: Optional[int] = None
    paper_id: str = ""
    theorem_index: int = 0
    quality: Optional[Dict] = None


class LLMInterface:
    """Abstract interface for different LLM backends."""
    
    def generate_proof(self, context: str) -> Tuple[str, Dict]:
        """
        Generate a proof given context.
        
        Returns:
            (generated_text, metadata)
        """
        raise NotImplementedError


class RunPodGPTOSSClient(LLMInterface):
    """GPT-OSS client using RunPod serverless inference."""
    
    def __init__(self, endpoint_url: str, api_key: str, model_name: str = "gpt-oss-120b"):
        """
        Initialize RunPod client.
        
        Args:
            endpoint_url: RunPod serverless endpoint URL
            api_key: RunPod API key
            model_name: Name of the model to use
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model_name = model_name
        
    def generate_proof(self, context: str) -> Tuple[str, Dict]:
        """Generate proof using RunPod GPT-OSS."""
        
        payload = {
            "input": {
                "prompt": context,
                "max_tokens": 2048,
                "temperature": 0.1,  # Low temperature for mathematical reasoning
                "top_p": 0.9,
                "stop": ["\\end{proof}", "QED", "∎"],  # Mathematical proof endings
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(self.endpoint_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            generation_time = time.time() - start_time
            
            # Extract generated text from RunPod response format
            if "output" in result and "choices" in result["output"]:
                generated_text = result["output"]["choices"][0]["text"]
            elif "output" in result and "text" in result["output"]:
                generated_text = result["output"]["text"]
            else:
                generated_text = str(result.get("output", ""))
            
            metadata = {
                "generation_time": generation_time,
                "model": self.model_name,
                "token_count": result.get("usage", {}).get("total_tokens", 0)
            }
            
            return generated_text, metadata
            
        except requests.exceptions.RequestException as e:
            return f"Error: RunPod request failed: {e}", {"error": str(e)}
        except Exception as e:
            return f"Error: Unexpected error: {e}", {"error": str(e)}
        
"""NOTE:
The following stray shell command line was previously (accidentally) inserted here:
    C:/ProgramData/anaconda3/Scripts/conda.exe run -p C:/Users/junho/.conda/envs/d2l --no-capture-output python proof_agent_setup.py --test-openai
It has been removed because it causes a SyntaxError inside the module. Do not embed shell commands directly in source files.
"""
class OpenAIClient(LLMInterface):
    """OpenAI API client for GPT-5 and other models."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-5, gpt-5-mini, gpt-4o, etc.)
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("OpenAI library not found. Install with: pip install openai")
    
    def set_reasoning_effort(self, effort: str):
        """Set reasoning effort (minimal|medium|high) for reasoning-capable models (e.g., gpt-5)."""
        if effort not in {"minimal", "medium", "high"}:
            raise ValueError("Invalid reasoning effort; choose from minimal, medium, high")
        self.reasoning_effort = effort

    def generate_proof(self, context: str) -> Tuple[str, Dict]:
        """Generate proof using OpenAI API (responses endpoint)."""
        start_time = time.time()
        try:
            # OpenAI new API style (Responses)
            full_prompt = (
                "System: You are an expert mathematician. Provide the most rigorous proof you can. "
                "If a complete proof is not currently derivable from the provided context, do NOT refuse. "
                "Instead, produce best-effort partial reasoning with explicit GAP lines (format: 'GAP: <missing step>'). "
                "State any necessary assumptions explicitly and continue reasoning after gaps when logically possible. "
                "Avoid generic disclaimers; always supply concrete mathematical progress.\n\nUser:\n" + context
            )
            kwargs = {
                "model": self.model,
                "input": full_prompt,
                "max_output_tokens": 3000,
                "temperature": 0.1,
                "top_p": 0.9,
            }
            # Optionally include reasoning effort if user set it and model likely supports it
            effort = getattr(self, "reasoning_effort", None)
            if effort:
                kwargs["reasoning"] = {"effort": effort}

            def _call(payload):
                return self.client.responses.create(**payload)

            response = None
            first_error = None
            try:
                response = _call(kwargs)
            except Exception as e:
                msg = str(e)
                first_error = msg
                unsupported = False
                # Detect unsupported parameter errors (temp/top_p) for some reasoning models
                if "Unsupported parameter" in msg and ("temperature" in msg or "top_p" in msg):
                    unsupported = True
                if unsupported:
                    minimal_kwargs = {k:v for k,v in kwargs.items() if k not in {"temperature","top_p"}}
                    try:
                        response = _call(minimal_kwargs)
                        minimal_used = True
                        kwargs = minimal_kwargs
                    except Exception as e2:
                        # Raise combined error
                        raise RuntimeError(f"Initial param error: {first_error}; Retry failed: {e2}")
                else:
                    raise

            generation_time = time.time() - start_time

            # Robust recursive extraction
            def _to_plain(obj):
                """Convert SDK objects to plain python for debug serialization."""
                import collections
                if obj is None:
                    return None
                # Prevent giant binary blobs (not expected here)
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                if isinstance(obj, dict):
                    return {k: _to_plain(v) for k, v in list(obj.items())[:100]}
                if isinstance(obj, (list, tuple, set)):
                    return [_to_plain(v) for v in list(obj)[:100]]
                # SDK objects often have dict() or model_dump()
                for attr in ("model_dump", "to_dict", "dict"):
                    if hasattr(obj, attr):
                        try:
                            data = getattr(obj, attr)()
                            return _to_plain(data)
                        except Exception:
                            pass
                # Fallback: select simple attributes
                simple = {}
                for name in dir(obj):
                    if name.startswith('_'):
                        continue
                    try:
                        val = getattr(obj, name)
                    except Exception:
                        continue
                    if isinstance(val, (str, int, float, bool)):
                        simple[name] = val
                return simple

            def _collect_text(obj, acc: List[str]):
                # Base primitives
                if obj is None:
                    return
                if isinstance(obj, str):
                    s = obj.strip()
                    if s:
                        acc.append(s)
                    return
                if isinstance(obj, (int, float, bool)):
                    return
                # dict-like
                if isinstance(obj, dict):
                    # Prefer explicit text fields first
                    for key in ["text", "output_text", "content"]:
                        if key in obj and isinstance(obj[key], str):
                            _collect_text(obj[key], acc)
                    # Recurse others
                    for v in obj.values():
                        _collect_text(v, acc)
                    return
                # list / tuple
                if isinstance(obj, (list, tuple, set)):
                    for item in obj:
                        _collect_text(item, acc)
                    return
                # SDK objects: inspect attributes
                for attr in ["text", "output_text", "content", "value"]:
                    if hasattr(obj, attr):
                        try:
                            val = getattr(obj, attr)
                        except Exception:
                            val = None
                        if isinstance(val, (str, list, dict)):
                            _collect_text(val, acc)
                # Fallback: if object has iterable output attribute
                if hasattr(obj, 'output'):
                    _collect_text(getattr(obj, 'output'), acc)

            text_chunks: List[str] = []
            _collect_text(response, text_chunks)

            # Track whether we performed an incomplete-output retry (avoid mutating API kwargs)
            retried_incomplete_reasoning = False

            # If we failed to extract any textual chunks and the response indicates
            # an incomplete reasoning-only output (common with GPT-5 when the
            # reasoning phase exhausts max_output_tokens before emitting final text),
            # attempt one controlled retry with a larger max_output_tokens budget
            # and stripped sampling parameters for maximum determinism.
            if not text_chunks:
                try:
                    resp_status = getattr(response, 'status', None)
                    incomplete_details = getattr(response, 'incomplete_details', None)
                    reason = None
                    if incomplete_details is not None:
                        # openai Responses SDK objects may expose dict-like access
                        if isinstance(incomplete_details, dict):
                            reason = incomplete_details.get('reason')
                        else:
                            reason = getattr(incomplete_details, 'reason', None)
                    # Only trigger retry if clearly max_output_tokens exhaustion
                    if resp_status == 'incomplete' and reason == 'max_output_tokens':
                        retry_kwargs = {k: v for k, v in kwargs.items() if k not in {'temperature', 'top_p'}}
                        current_max = retry_kwargs.get('max_output_tokens', 3000) or 3000
                        retry_kwargs['max_output_tokens'] = min(int(current_max * 2), 8000)
                        try:
                            retry_resp = _call(retry_kwargs)
                            response = retry_resp
                            kwargs = retry_kwargs  # adopt new params for metadata (already stripped)
                            text_chunks = []
                            _collect_text(response, text_chunks)
                            retried_incomplete_reasoning = True
                        except Exception as rerr:
                            text_chunks = []
                            text_chunks.append(f"(Retry after incomplete reasoning failed: {rerr})")
                except Exception:
                    pass
            # Filter out obvious JSON structural artifacts / very short tokens like '{', '}'
            cleaned = []
            for t in text_chunks:
                if len(t) < 2:
                    continue
                if t in {'{', '}', '[', ']', ':', ','}:
                    continue
                cleaned.append(t)
            # Heuristic: choose the longest contiguous chunk that looks like natural language proof
            if cleaned:
                # Merge consecutive lines that are plausible proof content
                # Strategy: pick the longest element > 40 chars containing spaces
                candidates = [c for c in cleaned if ' ' in c and len(c) > 40]
                if candidates:
                    # Sometimes model returns duplicated contexts; join unique long blocks sequentially
                    seen = set()
                    ordered = []
                    for c in candidates:
                        sig = c[:80]
                        if sig not in seen:
                            ordered.append(c)
                            seen.add(sig)
                    generated_text = "\n\n".join(ordered)
                else:
                    generated_text = max(cleaned, key=len)
            else:
                generated_text = "(No textual output extracted)"

            usage = getattr(response, 'usage', None)
            metadata = {
                "generation_time": generation_time,
                "model": self.model,
                "token_count": getattr(usage, 'total_tokens', None),
                "input_tokens": getattr(usage, 'input_tokens', None),
                "output_tokens": getattr(usage, 'output_tokens', None),
                "raw_response": _to_plain(response)
            }
            # Record if we stripped parameters
            if 'temperature' not in kwargs:
                metadata['stripped_temperature'] = True
            if 'top_p' not in kwargs:
                metadata['stripped_top_p'] = True
            if retried_incomplete_reasoning:
                metadata['retried_incomplete_reasoning'] = True
            if effort:
                metadata["reasoning_effort"] = effort
            return generated_text, metadata
        except Exception as e:
            msg = str(e)
            suggestion = None
            if 'Unsupported parameter' in msg:
                suggestion = "Remove extra parameters; try minimal call with only model and input."
            elif 'model' in msg.lower():
                suggestion = (
                    f"Model '{self.model}' may not be enabled. Try 'gpt-4o' or 'gpt-4o-mini'. Use --test-openai --model=gpt-4o-mini to verify entitlement."
                )
            error_payload = {"error": msg}
            if suggestion:
                error_payload["suggestion"] = suggestion
            return f"Error: OpenAI API request failed: {msg}\n" + (suggestion or ""), error_payload


class GeminiFlashClient(LLMInterface):
    """Gemini Flash API client as fallback option."""
    
    def __init__(self, api_key: str):
        """Initialize Gemini client."""
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
    def generate_proof(self, context: str) -> Tuple[str, Dict]:
        """Generate proof using Gemini Flash API."""
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": context
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 40,
                "topP": 0.9,
                "maxOutputTokens": 2048,
                "stopSequences": ["\\end{proof}", "QED", "∎"]
            }
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {"key": self.api_key}
        
        start_time = time.time()
        
        try:
            response = requests.post(
                self.base_url, 
                json=payload, 
                headers=headers, 
                params=params, 
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            generation_time = time.time() - start_time
            
            # Extract generated text from Gemini response
            if "candidates" in result and len(result["candidates"]) > 0:
                content = result["candidates"][0]["content"]
                if "parts" in content and len(content["parts"]) > 0:
                    generated_text = content["parts"][0]["text"]
                else:
                    generated_text = str(content)
            else:
                generated_text = "Error: No response generated"
            
            metadata = {
                "generation_time": generation_time,
                "model": "gemini-2.5-flash",
                "token_count": result.get("usageMetadata", {}).get("totalTokenCount", 0)
            }
            
            return generated_text, metadata
            
        except requests.exceptions.RequestException as e:
            return f"Error: Gemini API request failed: {e}", {"error": str(e)}
        except Exception as e:
            return f"Error: Unexpected error: {e}", {"error": str(e)}


class MockLLMClient(LLMInterface):
    """Mock client for testing without API costs."""
    
    def generate_proof(self, context: str) -> Tuple[str, Dict]:
        """Generate a mock proof for testing."""
        
        # Extract theorem statement for mock response
        theorem_match = context.find("**Theorem")
        if theorem_match == -1:
            theorem_match = context.find("**Lemma")
        
        mock_proof = """
**Proof**: We proceed by direct construction. Let $n$ be given as in the theorem statement.

First, we establish the necessary conditions. By the fundamental properties of the mathematical objects involved, we can apply the standard techniques from the relevant theory.

Consider the key observation that the statement follows from well-established results in this area. The construction can be made explicit by noting that the required properties are satisfied.

Therefore, the theorem holds as claimed. ∎

*Note: This is a mock proof generated for testing purposes.*
"""
        
        metadata = {
            "generation_time": 0.5,
            "model": "mock-llm",
            "token_count": 100
        }
        
        return mock_proof.strip(), metadata


class MathematicalProofAgent:
    """Agent for generating mathematical proofs using language models."""
    
    def __init__(self, llm_client: LLMInterface, results_dir: str = "proof_results",
                 rag_enabled: bool = False, rag_chunk_size: int = 900,
                 rag_overlap: int = 150, rag_top_k: int = 8):
        """
        Initialize proof agent.
        
        Args:
            llm_client: LLM client implementation
            results_dir: Directory to save proof results
        """
        self.llm_client = llm_client
        self.results_dir = results_dir
        self.data_manager = DataManager()
        self.context_preparator = ContextPreparator()
        # RAG configuration (can be overridden externally after instantiation)
        self.rag_enabled = rag_enabled
        self.rag_chunk_size = rag_chunk_size
        self.rag_overlap = rag_overlap
        self.rag_top_k = rag_top_k
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
    
    def generate_proof_for_theorem(self, paper_id: str, theorem_index: int, variant_tag: Optional[str] = None) -> ProofResult:
        """
        Generate a proof for a specific theorem in a paper.
        
        Args:
            paper_id: arXiv ID of the paper
            theorem_index: Index of the theorem to prove
            
        Returns:
            ProofResult with generated proof and metadata
        """
        print(f"\nGenerating proof for {paper_id}, theorem {theorem_index}...")
        
        # Load paper
        paper = self.data_manager.load_paper(paper_id)
        if not paper:
            return ProofResult(
                theorem_statement="",
                generated_proof="",
                model_used="",
                generation_time=0,
                success=False,
                error_message=f"Paper {paper_id} not found",
                paper_id=paper_id,
                theorem_index=theorem_index
            )
        
        try:
            # Prepare context
            # Support optional force_rag attribute (set externally via CLI)
            force_rag = getattr(self, 'force_rag', False)
            context = self.context_preparator.prepare_context(
                paper, theorem_index,
                rag_enabled=self.rag_enabled,
                rag_chunk_size=self.rag_chunk_size,
                rag_overlap=self.rag_overlap,
                rag_top_k=self.rag_top_k,
                force_rag=force_rag
            )
            provenance = self.context_preparator.get_last_provenance()
            
            # Format prompt for LLM
            prompt = self.context_preparator.format_for_llm_prompt(context)
            prompt_char_len = len(prompt)
            context_char_len = len(context.paper_context)
            
            print(f"Context prepared: {len(prompt):,} characters")
            print(f"Target theorem: {context.theorem_to_prove.statement_type.value}")
            
            # Generate proof
            start_time = time.time()
            generated_proof, metadata = self.llm_client.generate_proof(prompt)
            generation_time = time.time() - start_time
            
            # Create result
            result = ProofResult(
                theorem_statement=context.theorem_to_prove.statement,
                generated_proof=generated_proof,
                model_used=metadata.get("model", "unknown"),
                generation_time=generation_time,
                success=not generated_proof.startswith("Error:"),
                error_message=generated_proof if generated_proof.startswith("Error:") else None,
                token_count=metadata.get("token_count", 0),
                paper_id=paper_id,
                theorem_index=theorem_index
            )

            if result.success and result.generated_proof:
                result.quality = self._evaluate_proof_quality(result.generated_proof, result.theorem_statement)
                # Reclassify as failure if placeholder / too short unless it contains explicit GAP lines or a proof sketch section
                too_short = result.quality and result.quality.get('word_count', 0) < 30
                has_gap = 'GAP:' in result.generated_proof
                has_sections = '### Proof' in result.generated_proof or '### Proof Sketch' in result.generated_proof
                if result.generated_proof.strip() == "(No textual output extracted)" or (too_short and not has_gap and not has_sections):
                    result.success = False
                    if not result.error_message:
                        result.error_message = "LLM produced insufficient proof content (empty or too short)."
                # Attempt to split sketch vs full proof for later analysis
                sketch, full = self._split_proof_sections(result.generated_proof)
                if result.quality is not None:
                    result.quality['sketch_char_len'] = len(sketch) if sketch else 0
                    result.quality['full_proof_char_len'] = len(full) if full else 0
                    result.quality['has_split_proof_sections'] = bool(sketch or full)
                # If full section extracted, we can (optionally) replace generated_proof with unified format
                if full and sketch:
                    # Standardize storage layout for downstream consumers
                    combined = {
                        'proof_sketch': sketch.strip(),
                        'full_proof': full.strip()
                    }
                    try:
                        # Keep original text accessible but store structured JSON as string for readability
                        import json as _json
                        result.generated_proof = result.generated_proof  # keep original plain text in file
                        # We'll attach structured split in extra metadata via save function
                        result._split_structured = combined  # type: ignore
                    except Exception:
                        pass
            
            # Save result
            # Attach local token/context stats for persistence
            if not result.quality:
                result.quality = {}
            result.quality['prompt_char_len'] = prompt_char_len
            result.quality['context_char_len'] = context_char_len
            # Record provenance inside quality (lightweight)
            if provenance:
                result.quality['provenance'] = provenance
            # Structural validators (lightweight safety signals)
            try:
                structural = run_structural_validators(result.generated_proof, result.theorem_statement, provenance)
                result.quality['structural'] = structural
            except Exception as _sv_err:
                result.quality['structural_error'] = str(_sv_err)

            self._save_proof_result(result, extra_metadata=metadata, variant_tag=variant_tag)
            
            if result.success:
                print(f"[OK] Proof generated successfully ({result.generation_time:.2f}s)")
            else:
                print(f"[FAIL] Proof generation failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Error generating proof: {e}")
            return ProofResult(
                theorem_statement="",
                generated_proof="",
                model_used="",
                generation_time=0,
                success=False,
                error_message=str(e),
                paper_id=paper_id,
                theorem_index=theorem_index
            )
    
    def batch_generate_proofs(self, paper_id: str, max_theorems: int = 5) -> List[ProofResult]:
        """
        Generate proofs for multiple theorems in a paper.
        
        Args:
            paper_id: arXiv ID of the paper
            max_theorems: Maximum number of theorems to process
            
        Returns:
            List of ProofResult objects
        """
        print(f"\nBatch processing paper {paper_id} (max {max_theorems} theorems)...")
        
        # Load parsed content to count theorems
        parsed_file = f"data/parsed/{paper_id.replace('/', '_')}_parsed.json"
        if not os.path.exists(parsed_file):
            print(f"No parsed content found for {paper_id}")
            return []
        
        with open(parsed_file, 'r', encoding='utf-8') as f:
            parsed_data = json.load(f)
        
        theorems = parsed_data.get('theorems', [])
        num_theorems = min(len(theorems), max_theorems)
        
        print(f"Processing {num_theorems} theorems...")
        
        results = []
        for i in range(num_theorems):
            result = self.generate_proof_for_theorem(paper_id, i)
            results.append(result)
            
            # Brief pause between requests to be nice to APIs
            if i < num_theorems - 1:
                time.sleep(1)
        
        # Save batch summary
        self._save_batch_summary(paper_id, results)
        
        return results
    
    def _save_proof_result(self, result: ProofResult, extra_metadata: Optional[Dict] = None, variant_tag: Optional[str] = None):
        """Save individual proof result.

        Args:
            result: ProofResult dataclass instance
            extra_metadata: raw metadata returned by LLM client (may include raw_response)
        """
        filename_base = f"{result.paper_id.replace('/', '_')}_theorem_{result.theorem_index}"
        if variant_tag:
            filename_base += f"_{variant_tag}"
        filename = f"{filename_base}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # If structured split was added, surface it
        structured_split = getattr(result, '_split_structured', None)

        result_dict = {
            "paper_id": result.paper_id,
            "theorem_index": result.theorem_index,
            "theorem_statement": result.theorem_statement,
            "generated_proof": result.generated_proof,
            "proof_char_len": len(result.generated_proof) if result.generated_proof else 0,
            "proof_sections": structured_split if structured_split else None,
            "model_used": result.model_used,
            "generation_time": result.generation_time,
            "success": result.success,
            "error_message": result.error_message,
            "token_count": result.token_count,
            "generated_at": datetime.now().isoformat(),
            "quality": result.quality,
            "variant": variant_tag
        }
        
        # Attach select extra metadata (excluding potentially huge raw response) directly
        if extra_metadata:
            for k in ["input_tokens", "output_tokens"]:
                if k in extra_metadata and extra_metadata[k] is not None:
                    result_dict[k] = extra_metadata[k]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        # Persist raw_response separately if present
        if extra_metadata and 'raw_response' in extra_metadata:
            raw_path = os.path.join(self.results_dir, f"{filename_base}_raw_response.json")
            try:
                with open(raw_path, 'w', encoding='utf-8') as rf:
                    json.dump(extra_metadata['raw_response'], rf, indent=2, ensure_ascii=False)
                print(f"Raw response saved to {raw_path}")
            except Exception as e:
                print(f"Warning: Failed to save raw response: {e}")
        
        print(f"Saved result to {filepath}")

    def _split_proof_sections(self, proof_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Attempt to split the proof into sketch and full proof parts.

        Recognizes headings like '### Proof Sketch', '## Proof Sketch', '### Proof', '## Proof'.
        Falls back to heuristic detection: first heading containing 'Sketch' vs a later heading containing 'Proof'.

        Returns:
            (sketch_text, full_text) possibly None if not found.
        """
        if not proof_text:
            return None, None
        import re
        # Normalize line endings
        text = proof_text.replace('\r\n', '\n')
        # Primary regex for markdown headings
        heading_pattern = re.compile(r"^(#{2,4})\s*(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))
        sketch_span = None
        proof_span = None
        for i, m in enumerate(matches):
            heading = m.group(2).strip().lower()
            if sketch_span is None and 'sketch' in heading:
                # span end is start of next heading or end of text
                end = matches[i+1].start() if i+1 < len(matches) else len(text)
                sketch_span = (m.end(), end)
            if proof_span is None and (heading == 'proof' or heading.startswith('proof ')):
                end = matches[i+1].start() if i+1 < len(matches) else len(text)
                proof_span = (m.end(), end)
        # Fallback: look for markers inside the text if no markdown headings captured
        if sketch_span is None:
            alt = re.search(r"Proof Sketch:?\n", text, re.IGNORECASE)
            if alt:
                start = alt.end()
                # until '### Proof' or '## Proof' or end
                next_proof = re.search(r"^#{2,4}\s*Proof\b", text[start:], re.MULTILINE | re.IGNORECASE)
                end = start + next_proof.start() if next_proof else len(text)
                sketch_span = (start, end)
        if proof_span is None:
            alt2 = re.search(r"^#{2,4}\s*Proof\b", text, re.MULTILINE | re.IGNORECASE)
            if alt2:
                start = alt2.end()
                proof_span = (start, len(text))
        # Extract slices
        sketch_text = text[sketch_span[0]:sketch_span[1]].strip() if sketch_span else None
        full_text = text[proof_span[0]:proof_span[1]].strip() if proof_span else None
        # Basic sanity: avoid returning identical large blocks
        if sketch_text and full_text and sketch_text == full_text:
            # If identical, treat as no split
            return None, full_text
        return sketch_text, full_text
    
    def _save_batch_summary(self, paper_id: str, results: List[ProofResult]):
        """Save batch processing summary."""
        filename = f"{paper_id.replace('/', '_')}_batch_summary.json"
        filepath = os.path.join(self.results_dir, filename)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        summary = {
            "paper_id": paper_id,
            "total_theorems": len(results),
            "successful_proofs": len(successful),
            "failed_proofs": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_time": sum(r.generation_time for r in results),
            "total_tokens": sum(r.token_count or 0 for r in results),
            "model_used": results[0].model_used if results else "",
            "generated_at": datetime.now().isoformat(),
            "results": [
                {
                    "theorem_index": r.theorem_index,
                    "success": r.success,
                    "generation_time": r.generation_time,
                    "token_count": r.token_count,
                    "error": r.error_message,
                    "quality": r.quality
                } for r in results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Batch summary saved to {filepath}")
        print(f"Success rate: {summary['success_rate']:.1%} ({len(successful)}/{len(results)})")

    def _evaluate_proof_quality(self, proof_text: str, theorem_text: str) -> Dict:
        """Lightweight heuristic scoring for triage (not formal verification)."""
        quality: Dict[str, Optional[object]] = {}
        dollars = proof_text.count('$')
        quality['math_balance_ok'] = (dollars % 2 == 0)
        markers = ["Therefore", "Thus", "Hence", "It follows", "Consequently"]
        structure_hits = sum(1 for m in markers if m.lower() in proof_text.lower())
        quality['structure_markers'] = structure_hits
        conclusion_present = any(sym in proof_text for sym in ["∎", "QED", "Q.E.D", "□"])
        quality['conclusion_symbol'] = conclusion_present
        words = proof_text.split()
        wc = len(words)
        quality['word_count'] = wc
        quality['length_category'] = 'short' if wc < 120 else ('medium' if wc < 400 else 'long')
        lemma_refs = re.findall(r"Lemma\s+\d+[A-Za-z0-9\.]*", proof_text)
        quality['lemma_ref_count'] = len(lemma_refs)
        def tokenize(t: str):
            return {w.lower() for w in re.findall(r"[A-Za-z0-9_]+", t)}
        theorem_tokens = tokenize(theorem_text)
        proof_tokens = tokenize(proof_text)
        overlap = len(theorem_tokens & proof_tokens) / len(theorem_tokens) if theorem_tokens else 0
        quality['theorem_overlap_ratio'] = round(overlap, 3)
        aggregate = 0
        if quality['math_balance_ok']:
            aggregate += 1
        if conclusion_present:
            aggregate += 1
        aggregate += min(structure_hits, 2)
        if quality['lemma_ref_count'] > 5:
            aggregate -= 1
        if quality['length_category'] == 'medium':
            aggregate += 1
        quality['aggregate_score'] = aggregate
        return quality


def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii','replace').decode())

def main():
    """CLI entry: supports single theorem or batch processing.
    Usage examples:
      python proof_agent.py --model=gpt-4o --paper=2509.22618 --theorem=0
      python proof_agent.py --model=gpt-4o --paper=2509.22618 --batch --limit=5
      python proof_agent.py --mock (forces mock model)
    Defaults: first saved paper, theorem 0, mock model if no model specified.
    """
    import sys
    args = sys.argv[1:]

    # Parse args
    def get_arg(prefix: str, default=None):
        for a in args:
            if a.startswith(prefix):
                return a.split('=',1)[1]
        return default

    use_batch = '--batch' in args
    force_mock = '--mock' in args
    limit = int(get_arg('--limit=', '5'))
    model = get_arg('--model=', None)
    paper_override = get_arg('--paper=', None)
    theorem_idx = int(get_arg('--theorem=', '0'))

    print("Mathematical Proof Agent")
    print("="*50)

    # Select LLM client
    if force_mock:
        print("Using mock LLM client (override)")
        llm_client = MockLLMClient()
    else:
        if model:
            try:
                from proof_agent_setup import ProofAgentConfig
                llm_client = ProofAgentConfig.create_openai_client(model=model)
                print(f"Using OpenAI model: {model}")
            except Exception as e:
                print(f"Failed to initialize OpenAI model '{model}': {e}. Falling back to mock.")
                llm_client = MockLLMClient()
        else:
            print("No --model specified; defaulting to mock client. Use --model=gpt-4o to enable API.")
            llm_client = MockLLMClient()

    agent = MathematicalProofAgent(llm_client)
    data_manager = DataManager()
    papers = data_manager.list_saved_papers()
    if not papers:
        print("No papers found. Run the data pipeline first.")
        return

    # Resolve paper
    if paper_override:
        paper_id = paper_override
    else:
        paper_id = papers[0]['arxiv_id']

    if use_batch:
        print(f"Batch mode: paper={paper_id} limit={limit}")
        results = agent.batch_generate_proofs(paper_id, max_theorems=limit)
        # Print brief summary
        successes = sum(1 for r in results if r.success)
        print(f"Completed batch: {successes}/{len(results)} successful")
        # Show quality aggregate if available
        qualities = [r.quality for r in results if r.quality]
        if qualities:
            avg_score = sum(q['aggregate_score'] for q in qualities)/len(qualities)
            print(f"Average aggregate quality score: {avg_score:.2f}")
        return
    else:
        print(f"Single theorem mode: paper={paper_id} theorem={theorem_idx}")
        result = agent.generate_proof_for_theorem(paper_id, theorem_idx)
        print("\nResult: success=" + str(result.success))
        if result.quality:
            print("Quality aggregate score:", result.quality.get('aggregate_score'))
        return


if __name__ == "__main__":
    main()