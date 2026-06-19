import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_pipeline import get_dataloaders
import numpy as np
from tqdm import tqdm

def evaluate_model(model, dataloader, device, desc="Evaluating"):
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            # Limit evaluation steps for speed
            if total_steps > 20: break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
            
    avg_loss = total_loss / total_steps if total_steps > 0 else 0
    perplexity = np.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, perplexity

def run_full_evaluation(adapter_path=None, model_name="gpt2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Loading Model for Evaluation: {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    if adapter_path:
        print(f"Loading LoRA Adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("Using Base Model (Zero-Shot)")
        model = base_model
        
    model.to(device)
    
    factual_loader, adaptation_loader = get_dataloaders(tokenizer, batch_size=2)
    
    print("\n--- Evaluating Factual Retention (TruthfulQA) ---")
    f_loss, f_ppl = evaluate_model(model, factual_loader, device, desc="TruthfulQA")
    
    print("\n--- Evaluating Adaptation Quality (PubMedQA) ---")
    a_loss, a_ppl = evaluate_model(model, adaptation_loader, device, desc="PubMedQA")
    
    return {
        "factual_loss": f_loss,
        "factual_ppl": f_ppl,
        "adaptation_loss": a_loss,
        "adaptation_ppl": a_ppl
    }

if __name__ == "__main__":
    import sys
    import json
    
    adapter = sys.argv[1] if len(sys.argv) > 1 else None
    results = run_full_evaluation(adapter_path=adapter)
    
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(json.dumps(results, indent=4))
    
    # Save to file
    out_name = f"results_{adapter.replace('./', '').replace('/', '_') if adapter else 'base'}.json"
    with open(out_name, "w") as f:
        json.dump(results, f)
