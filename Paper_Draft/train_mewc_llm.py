import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from mewc_lora_optimizer import MEWCLoRAOptimizer
from data_pipeline import get_dataloaders

def setup_model(model_name="gpt2"):
    """
    Sets up a base LLM and attaches a LoRA adapter for parameter-efficient fine-tuning.
    """
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (use bf16 or int8 if memory is limited)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    print("Injecting LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_mewc_llm(method="mewc"):
    """
    Simulates the MEWC-LLM continual learning pipeline or baseline LoRA.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = setup_model("gpt2") 
    model.to(device)
    
    if method == "mewc":
        print("\n--- Method: MEWC-LLM (Conflict-Aware) ---")
        optimizer = MEWCLoRAOptimizer(
            model.parameters(), 
            lr=5e-5, 
            conflict_threshold=0.0,
            penalty_weight=100.0
        )
    else:
        print("\n--- Method: Baseline LoRA (Standard AdamW) ---")
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=5e-5)
    
    print("\n--- Phase 0: Data Acquisition ---")
    factual_dataloader, adaptation_dataloader = get_dataloaders(tokenizer, batch_size=2)
    
    if method == "mewc":
        print("\n--- Phase 1: Factual Consolidation (MEWC only) ---")
        print("Computing Fisher Information and Factual Gradients...")
        optimizer.update_factual_memory(model, factual_dataloader, device)
        print("Factual Memory Consolidated.\n")
    
    print("--- Phase 2: Continual Adaptation ---")
    print(f"Training on new domain data using {method}...")
    
    model.train()
    epochs = 1 # Reducing to 1 epoch for faster iteration in results gathering
    
    for epoch in range(epochs):
        for step, batch in enumerate(adaptation_dataloader):
            # Limit to 5 steps for demonstration results
            if step > 5: break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {step} | Loss: {loss.item():.4f}")
    
    save_path = f"./{method}_lora_weights"
    print(f"\nTraining Complete! Saving weights to {save_path}...")
    model.save_pretrained(save_path)

if __name__ == "__main__":
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "mewc"
    train_mewc_llm(method=method)
