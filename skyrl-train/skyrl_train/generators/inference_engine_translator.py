from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import asyncio
import uuid
import time
from skyrl_train.inference_engines.base import InferenceEngineInput
from dataclasses import dataclass

# Pydantic models for OpenAI API compatibility
class CompletionRequest(BaseModel):
    prompt: Union[str, List[str]] = Field(..., description="The prompt(s) to generate completions for")
    model: Optional[str] = Field(default="your-model", description="Model to use")
    max_tokens: Optional[int] = Field(default=100, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    n: Optional[int] = Field(default=1, ge=1, le=10, description="Number of completions to generate")
    stream: Optional[bool] = Field(default=False, description="Whether to stream results")
    logprobs: Optional[int] = Field(default=None, ge=0, le=5, description="Include log probabilities")
    echo: Optional[bool] = Field(default=False, description="Echo back the prompt")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(default=1, ge=1, le=20)
    logit_bias: Optional[Dict[str, float]] = Field(default=None)
    user: Optional[str] = Field(default=None, description="User identifier")

class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage



# Adapter class to wrap your inference engine
class InferenceEngineAdapter:
    def __init__(self, inference_engine_client, tokenizer, use_conversation_multi_turn=True):
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer
        self.use_conversation_multi_turn = use_conversation_multi_turn
    
    def _create_sampling_params(self, request: CompletionRequest):
        """Convert OpenAI parameters to your sampling params format"""
        # You'll need to adapt this based on your actual sampling_params structure
        return {
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'stop': request.stop,
            'presence_penalty': request.presence_penalty,
            'frequency_penalty': request.frequency_penalty,
            # Add other parameters as needed for your system
        }
    
    def _prepare_input(self, prompt: str, trajectory_id: str, sampling_params) -> InferenceEngineInput:
        """Prepare the input for your inference engine"""
        if self.use_conversation_multi_turn:
            # Convert prompt to chat_history format - you may need to adjust this
            chat_history = [{"role": "user", "content": prompt}]
            return InferenceEngineInput(
                prompts=[chat_history], 
                trajectory_ids=[trajectory_id], 
                sampling_params=sampling_params
            )
        else:
            # Convert prompt to token IDs - you'll need your tokenization logic here
            input_ids = self._tokenize(prompt)  # Implement this method
            return InferenceEngineInput(
                prompt_token_ids=[input_ids], 
                trajectory_ids=[trajectory_id], 
                sampling_params=sampling_params
            )
    
    def _tokenize(self, prompt: str):
        """Tokenize the prompt using the provided tokenizer"""
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(prompt)
        elif hasattr(self.tokenizer, 'tokenize'):
            return self.tokenizer.tokenize(prompt)
        else:
            # Fallback if tokenizer has different interface
            return self.tokenizer(prompt)
    
    def _count_tokens(self, text: str) -> int:
        """Accurately count tokens using the tokenizer"""
        try:
            tokens = self._tokenize(text)
            return len(tokens)
        except Exception:
            # Fallback to rough estimation if tokenization fails
            return int(len(text.split()) * 1.3)
    
    def _estimate_tokens(self, text: str) -> int:
        """Accurate token counting - now uses the tokenizer"""
        return self._count_tokens(text)
    
    async def generate_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion using your inference engine"""
        trajectory_id = str(uuid.uuid4())
        sampling_params = self._create_sampling_params(request)
        
        # Handle single prompt
        prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
        
        try:
            # Prepare input for your inference engine
            engine_input = self._prepare_input(prompt, trajectory_id, sampling_params)
            
            # Call your inference engine
            engine_output = await self.inference_engine_client.generate(engine_input)
            
            # Extract the generated text from your engine output
            # You'll need to adapt this based on your engine_output structure
            generated_text = self._extract_generated_text(engine_output)
            
            # Create OpenAI-compatible response
            choices = [
                CompletionChoice(
                    text=generated_text,
                    index=0,
                    finish_reason="stop"  # or "length" based on your logic
                )
            ]
            
            # Calculate token usage
            prompt_tokens = int(self._estimate_tokens(prompt))
            completion_tokens = int(self._estimate_tokens(generated_text))
            
            usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=choices,
                usage=usage
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, f"Inference engine error: {str(e)}")
    
    def _extract_generated_text(self, engine_output) -> str:
        """Extract generated text from your engine output"""
        # You'll need to implement this based on your engine_output structure
        # This is a placeholder
        if hasattr(engine_output, 'text'):
            return engine_output.text
        elif hasattr(engine_output, 'outputs') and engine_output.outputs:
            return engine_output.outputs[0].text if hasattr(engine_output.outputs[0], 'text') else str(engine_output.outputs[0])
        else:
            return str(engine_output)

# FastAPI app
app = FastAPI(title="OpenAI Compatible Completions API")

# Initialize your adapter (you'll need to pass your actual inference_engine_client)
# adapter = InferenceEngineAdapter(your_inference_engine_client)

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """OpenAI compatible completions endpoint"""
    if not hasattr(app.state, 'adapter'):
        raise HTTPException(status_code=500, detail="Inference engine not initialized")
    
    try:
        return await app.state.adapter.generate_completion(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "your-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "your-organization"
            }
        ]
    }

# Initialization function
def initialize_server(inference_engine_client, tokenizer, use_conversation_multi_turn=True):
    """Initialize the server with your inference engine client and tokenizer"""
    app.state.adapter = InferenceEngineAdapter(
        inference_engine_client, 
        tokenizer,
        use_conversation_multi_turn
    )
    return app

# Example usage:
if __name__ == "__main__":
    import uvicorn
    
    # You would initialize with your actual inference_engine_client
    # your_inference_engine_client = YourInferenceEngineClient()
    # app = initialize_server(your_inference_engine_client)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Example client usage:
"""
import requests

response = requests.post("http://localhost:8000/v1/completions", json={
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
})

print(response.json())
"""