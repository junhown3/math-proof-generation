"""
Configuration and setup for Mathematical Proof Agent
Handles different LLM backends and API keys.
"""

import os
from proof_agent import (
    MathematicalProofAgent, 
    OpenAIClient,
    RunPodGPTOSSClient, 
    GeminiFlashClient, 
    MockLLMClient
)


class ProofAgentConfig:
    """Configuration manager for the proof agent."""
    
    @staticmethod
    def create_openai_client(api_key: str = None, model: str = "gpt-4o") -> OpenAIClient:
        """
        Create OpenAI API client.
        
        Setup instructions:
        1. Go to OpenAI Platform: https://platform.openai.com/
        2. Create an API key in your account settings
        3. Set OPENAI_API_KEY environment variable or pass directly
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-5, gpt-5-mini, gpt-4o, etc.)
        """
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass it directly."
            )
        
        return OpenAIClient(api_key, model)
    
    @staticmethod
    def create_runpod_client(endpoint_url: str = None, api_key: str = None) -> RunPodGPTOSSClient:
        """
        Create RunPod GPT-OSS client.
        
        Setup instructions:
        1. Go to RunPod console: https://console.runpod.io/
        2. Create a serverless endpoint with GPT-OSS model
        3. Get your endpoint URL and API key
        4. Set environment variables or pass directly
        
        Args:
            endpoint_url: RunPod serverless endpoint URL
            api_key: RunPod API key
        """
        endpoint_url = endpoint_url or os.getenv('RUNPOD_ENDPOINT_URL')
        api_key = api_key or os.getenv('RUNPOD_API_KEY')
        
        if not endpoint_url or not api_key:
            raise ValueError(
                "RunPod credentials required. Set RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY "
                "environment variables or pass them directly."
            )
        
        return RunPodGPTOSSClient(endpoint_url, api_key)
    
    @staticmethod
    def create_gemini_client(api_key: str = None) -> GeminiFlashClient:
        """
        Create Gemini Flash client.
        
        Setup instructions:
        1. Go to Google AI Studio: https://ai.google.dev/
        2. Create a new API key
        3. Set GEMINI_API_KEY environment variable or pass directly
        
        Args:
            api_key: Gemini API key
        """
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass it directly."
            )
        
        return GeminiFlashClient(api_key)
    
    @staticmethod
    def create_mock_client() -> MockLLMClient:
        """Create mock client for testing."""
        return MockLLMClient()


def setup_openai_instructions():
    """Print OpenAI API setup instructions."""
    print("""
🚀 OPENAI GPT-5 SETUP INSTRUCTIONS

1. Get API Key:
   - Go to OpenAI Platform: https://platform.openai.com/
   - Sign in or create an account
   - Go to API Keys: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy your API key (starts with sk-...)
   
2. Set Environment Variable (Windows PowerShell):
   $env:OPENAI_API_KEY="sk-your_api_key_here"
   
3. Add Credits:
   - Go to Billing: https://platform.openai.com/settings/organization/billing
   - Add payment method and credits ($10-20 should be plenty for testing)
   
4. Test Connection:
   python proof_agent_setup.py --test-openai

💰 Expected Costs (GPT-5):
- Input: $1.25 per 1M tokens
- Output: $10.00 per 1M tokens  
- For 440 theorems (~24M tokens): ~$273 total
- For testing (10 theorems): ~$6

🎯 Available Models:
- gpt-5: Best reasoning (recommended)
- gpt-5-mini: Faster and cheaper
- gpt-4o: Previous generation but still excellent

🔥 Why GPT-5?
- State-of-the-art mathematical reasoning
- Excellent at formal proofs
- Large context window
- Immediate availability (no setup)
""")


def setup_runpod_instructions():
    """Print detailed RunPod setup instructions."""
    print("""
🚀 RUNPOD SETUP INSTRUCTIONS FOR GPT-OSS

1. Create RunPod Account:
   - Go to https://runpod.io/
   - Sign up and add credits to your account
   
2. Deploy GPT-OSS Model:
   - Go to RunPod Console: https://console.runpod.io/
   - Click "Serverless" → "New Endpoint"
   - Search for "GPT-OSS" or "OpenAI GPT-OSS 120B" template
   - Configure:
     * GPU: A100 80GB or H100 80GB
     * Max workers: 1-3 (depending on usage)
     * Timeout: 300 seconds
   
3. Get Credentials:
   - After deployment, note your:
     * Endpoint URL (looks like: https://api.runpod.ai/v2/xxx/runsync)
     * API Key (in account settings)
   
4. Set Environment Variables (Windows PowerShell):
   $env:RUNPOD_ENDPOINT_URL="your_endpoint_url_here"
   $env:RUNPOD_API_KEY="your_api_key_here"
   
5. Test Connection:
   python proof_agent_setup.py --test-runpod

💰 Expected Costs:
- A100 80GB: ~$1.19/hour
- H100 80GB: ~$1.99/hour  
- For 440 theorems: ~$36-70 total

📚 Alternative Models:
If GPT-OSS 120B isn't available, try:
- Meta Llama 3.1 70B
- DeepSeek Math 7B
- Qwen2.5 Math 72B
""")


def setup_gemini_instructions():
    """Print Gemini API setup instructions."""
    print("""
💎 GEMINI FLASH SETUP INSTRUCTIONS

1. Get API Key:
   - Go to Google AI Studio: https://ai.google.dev/
   - Click "Get API key" → "Create API key"
   - Copy your API key
   
2. Set Environment Variable (Windows PowerShell):
   $env:GEMINI_API_KEY="your_api_key_here"
   
3. Test Connection:
   python proof_agent_setup.py --test-gemini

💰 Expected Costs:
- Gemini 2.5 Flash: $0.30 input + $2.50 output per 1M tokens
- For 440 theorems (~24M tokens): ~$66 total

🎯 Why Gemini Flash?
- Excellent mathematical reasoning
- Much cheaper than GPT-5/Gemini Pro
- Fast inference speed
- Large context window (1M tokens)
""")


def main():
    """Setup assistant for proof agent."""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--setup-openai":
            setup_openai_instructions()
        elif command == "--setup-runpod":
            setup_runpod_instructions()
        elif command == "--setup-gemini":
            setup_gemini_instructions()
        elif command == "--test-openai":
            try:
                # Optional: allow model override via --model=<name>
                override_model = None
                for arg in sys.argv[2:]:
                    if arg.startswith("--model="):
                        override_model = arg.split("=",1)[1]
                client = ProofAgentConfig.create_openai_client(model=override_model or "gpt-4o")
                print("✓ OpenAI client created successfully!")
                print("Testing inference (short sanity prompt)...")
                result, metadata = client.generate_proof("Briefly confirm: 1+1=2. Then stop.")
                if result.startswith("Error:"):
                    print("✗ OpenAI inference error returned:")
                    print(result[:400])
                    if metadata.get('error'):
                        print(f"Metadata error: {metadata['error']}")
                        if 'suggestion' in metadata:
                            print(f"Suggestion: {metadata['suggestion']}")
                else:
                    snippet = result[:160].replace('\n',' ') + ("..." if len(result)>160 else "")
                    print(f"✓ OpenAI inference working: {snippet}")
                if metadata.get('model'):
                    print(f"Model: {metadata.get('model')}  Tokens: {metadata.get('token_count')}  Input: {metadata.get('input_tokens')}  Output: {metadata.get('output_tokens')}")
                else:
                    print("(No model metadata returned; call likely failed upstream.)")
            except Exception as e:
                print(f"✗ OpenAI setup failed: {e}")
        elif command == "--test-runpod":
            try:
                client = ProofAgentConfig.create_runpod_client()
                print("✓ RunPod client created successfully!")
                print("Testing inference...")
                result, metadata = client.generate_proof("Test prompt: What is 2+2?")
                print(f"✓ RunPod inference working: {result[:100]}...")
            except Exception as e:
                print(f"✗ RunPod setup failed: {e}")
        elif command == "--test-gemini":
            try:
                client = ProofAgentConfig.create_gemini_client()
                print("✓ Gemini client created successfully!")
                print("Testing inference...")
                result, metadata = client.generate_proof("Test prompt: What is 2+2?")
                print(f"✓ Gemini inference working: {result[:100]}...")
            except Exception as e:
                print(f"✗ Gemini setup failed: {e}")
        else:
            print("Unknown command. Use --setup-runpod, --setup-gemini, --test-runpod, or --test-gemini")
    else:
        print("""
Mathematical Proof Agent Setup

Available commands:
  python proof_agent_setup.py --setup-openai     # OpenAI GPT-5 setup instructions
  python proof_agent_setup.py --setup-runpod     # RunPod setup instructions
  python proof_agent_setup.py --setup-gemini     # Gemini setup instructions  
  python proof_agent_setup.py --test-openai      # Test OpenAI connection
  python proof_agent_setup.py --test-runpod      # Test RunPod connection
  python proof_agent_setup.py --test-gemini      # Test Gemini connection

For quick start with mock testing:
  python proof_agent.py

Recommended setup order for quick start:
1. Set up OpenAI GPT-5 for immediate high-quality results
2. Test with a few theorems to validate approach
3. Switch to RunPod GPU for large-scale cost-effective processing

Recommended setup order for research:
1. Start with mock testing to verify the pipeline
2. Set up RunPod for cost-effective GPU inference  
3. Use OpenAI/Gemini for comparison and validation
""")


if __name__ == "__main__":
    main()