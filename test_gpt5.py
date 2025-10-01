"""
Quick Start: Test OpenAI Mathematical Proof Generation
Supports selectable model via CLI argument: --model=gpt-4o (default), gpt-4o-mini, gpt-5 (if enabled)
"""

from proof_agent import MathematicalProofAgent
from proof_agent_setup import ProofAgentConfig
from data_manager import DataManager
import sys


def parse_args_model(default_model="gpt-4o"):
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            return arg.split("=",1)[1]
    return default_model


def test_gpt5_proof_generation():
    model = parse_args_model()
    print(f"🚀 Testing Mathematical Proof Generation (model={model})")
    print("=" * 50)
    
    try:
        print("Creating client...")
        client = ProofAgentConfig.create_openai_client(model=model)
        
        # Initialize agent
        agent = MathematicalProofAgent(client)
        
        # Get available papers
        data_manager = DataManager()
        papers = data_manager.list_saved_papers()
        
        if not papers:
            print("❌ No papers found. Run data_manager.py first to fetch papers.")
            return
        
        # Test with first paper, first theorem
        paper = papers[0]
        paper_id = paper['arxiv_id']
        
        print(f"\n📄 Testing with paper: {paper['title'][:60]}...")
        print(f"📋 Paper ID: {paper_id}")
        
        # Generate proof for first theorem
        print(f"\n🤖 Generating proof with GPT-5...")
        result = agent.generate_proof_for_theorem(paper_id, 0)
        
        # Display results
        print(f"\n{'='*60}")
        print("🎯 PROOF GENERATION RESULT")
        print(f"{'='*60}")
        
        print(f"✅ Success: {result.success}")
        print(f"🤖 Model: {result.model_used}")
        print(f"⏱️  Time: {result.generation_time:.2f} seconds")
        print(f"🔢 Tokens: {result.token_count:,}")
        
        if result.success:
            print(f"\n📝 Theorem Statement:")
            theorem = result.theorem_statement[:300]
            if len(result.theorem_statement) > 300:
                theorem += "..."
            print(theorem)
            
            print(f"\n🧠 Generated Proof:")
            proof = result.generated_proof[:800]
            if len(result.generated_proof) > 800:
                proof += "..."
            print(proof)
            
            input_tokens = getattr(result, 'input_tokens', None)
            output_tokens = getattr(result, 'output_tokens', None)
            total_tokens = result.token_count or ( (input_tokens or 0) + (output_tokens or 0) )
            # Display cost estimate only if we have at least total tokens
            if total_tokens:
                # Use tiered pricing heuristics (can refine per model later)
                ip = input_tokens or total_tokens//2
                op = output_tokens or total_tokens - ip
                est_cost = (ip/1_000_000*1.25) + (op/1_000_000*10.0)
                print(f"\n💰 Estimated Cost: ${est_cost:.4f} (input≈{ip} output≈{op})")
            else:
                print("\n💰 Cost: (token usage not returned by API)")
            
            if getattr(result, 'quality', None):
                print("\n🔍 Quality Heuristic:")
                for k,v in result.quality.items():
                    print(f"  {k}: {v}")
        
        else:
            print(f"\n❌ Error: {result.error_message}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Setup Error: {e}")
        print("\n💡 Make sure you have:")
        print("1. Set your OpenAI API key: $env:OPENAI_API_KEY=\"sk-...\"")
        print("2. Added billing to your OpenAI account")
        print("3. Have credits available")
        return None


def batch_test_gpt5(num_theorems: int = 3):
    model = parse_args_model()
    print(f"\n🚀 Batch Testing (model={model}, {num_theorems} theorems)")
    print("=" * 50)
    
    try:
        client = ProofAgentConfig.create_openai_client(model=model)
        agent = MathematicalProofAgent(client)
        
        # Get first paper
        data_manager = DataManager()
        papers = data_manager.list_saved_papers()
        paper_id = papers[0]['arxiv_id']
        
        print(f"📄 Processing paper: {papers[0]['title'][:50]}...")
        
        # Generate multiple proofs
        results = agent.batch_generate_proofs(paper_id, max_theorems=num_theorems)
        
        # Summary
        successful = [r for r in results if r.success]
        total_cost = 0
        total_tokens = sum(r.token_count for r in results if r.token_count)
        
        for result in results:
            if getattr(result, 'success', False) and getattr(result, 'token_count', None):
                # Rough split if detailed tokens missing
                ip = getattr(result, 'input_tokens', result.token_count//2 if result.token_count else 0)
                op = getattr(result, 'output_tokens', result.token_count - ip if result.token_count else 0)
                total_cost += (ip/1_000_000*1.25) + (op/1_000_000*10.0)
        
        print(f"\n📊 BATCH RESULTS")
        print(f"{'='*40}")
        print(f"✅ Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"⏱️  Total Time: {sum(r.generation_time for r in results):.2f} seconds")
        print(f"🔢 Total Tokens: {total_tokens:,}")
        print(f"💰 Estimated Total Cost: ${total_cost:.4f}")
        
        for result in results:
            if getattr(result, 'quality', None):
                print(f"Theorem {result.theorem_index} quality score: {result.quality.get('aggregate_score')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


if __name__ == "__main__":
    # Allow skipping interactive prompt if --auto provided
    auto = any(a == "--auto" for a in sys.argv[1:])
    if auto:
        test_gpt5_proof_generation()
    else:
        print("GPT-5 Mathematical Proof Generation Test")
        print("Choose an option:")
        print("1. Single theorem test")  
        print("2. Batch test (3 theorems)")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            test_gpt5_proof_generation()
        elif choice == "2":
            batch_test_gpt5(3)
        else:
            print("Testing single theorem by default...")
            test_gpt5_proof_generation()