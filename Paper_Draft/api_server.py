from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

app = FastAPI(title="MEWC-LLM Factual Inference API", 
              description="Real-world deployment of MEWC-protected LoRA LLM.")

# Global variables to hold model and tokenizer
model = None
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print("Loading MEWC-protected LLM for inference...")
    
    try:
        # Path where the MEWC LoRA weights are saved
        peft_model_id = "./mewc_lora_weights"
        config = PeftConfig.from_pretrained(peft_model_id)
        
        # Load base tokenizer and model
        base_model_id = config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
        
        # Merge the base model with the MEWC-protected LoRA adapter
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model from ./mewc_lora_weights. Did you run train_mewc_llm.py first? Error: {e}")

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    device = model.device
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"prompt": request.prompt, "generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
