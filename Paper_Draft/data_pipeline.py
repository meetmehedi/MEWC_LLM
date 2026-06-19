import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

class MEWCDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, features):
        batch = self.tokenizer(
            features,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # For causal LM, labels are the input_ids
        batch["labels"] = batch["input_ids"].clone()
        return batch

def format_truthful_qa(example):
    # Formats the TruthfulQA dataset into a conversational string
    question = example['question']
    best_answer = example['best_answer']
    return f"Question: {question}\nCorrect Answer: {best_answer}"

def format_pubmed_qa(example):
    # Formats the PubMedQA dataset into a conversational string
    question = example['question']
    context = "".join(example['context']['contexts'])
    answer = example['long_answer']
    return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"

def get_dataloaders(tokenizer, batch_size=4):
    """
    Downloads and prepares the Factual (TruthfulQA) and Adaptation (PubMedQA) dataloaders.
    """
    print("Downloading TruthfulQA (Factual Memory)...")
    # Load TruthfulQA generation split
    truth_dataset = load_dataset("truthful_qa", "generation", split="validation")
    
    # Format and collate
    formatted_truth = [format_truthful_qa(ex) for ex in truth_dataset]
    collator = MEWCDataCollator(tokenizer)
    
    factual_dataloader = DataLoader(
        formatted_truth, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collator
    )
    
    print("Downloading PubMedQA (New Domain Adaptation)...")
    # Load PubMedQA (medical domain)
    medical_dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # Take a subset for faster training demonstration
    medical_dataset = medical_dataset.select(range(min(1000, len(medical_dataset))))
    formatted_medical = [format_pubmed_qa(ex) for ex in medical_dataset]
    
    adaptation_dataloader = DataLoader(
        formatted_medical, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collator
    )
    
    return factual_dataloader, adaptation_dataloader

if __name__ == "__main__":
    # Test script locally
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    f_loader, a_loader = get_dataloaders(tokenizer, batch_size=2)
    print(f"Factual batches: {len(f_loader)}")
    print(f"Adaptation batches: {len(a_loader)}")
    
    # Print a sample
    for batch in f_loader:
        print("\nSample Factual Batch Shapes:")
        print(batch["input_ids"].shape)
        break
