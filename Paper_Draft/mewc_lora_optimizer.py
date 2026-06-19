import torch
from torch.optim import AdamW

class MEWCLoRAOptimizer(AdamW):
    """
    MEWC-LoRA Optimizer: Truth-Aware Adaptive Consolidation for LLMs.
    
    This optimizer extends AdamW to implement:
    1. Gradient Alignment Conflict Detection (Cosine Similarity)
    2. Fisher-Based Truth Importance Penalty
    
    It expects that `factual_gradients` and `fisher_information` are stored
    in the optimizer's state for the protected (LoRA) parameters.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, 
                 conflict_threshold=0.0, penalty_weight=100.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                         weight_decay=weight_decay, amsgrad=amsgrad)
        
        self.conflict_threshold = conflict_threshold
        self.penalty_weight = penalty_weight

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with adaptive consolidation.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Check if this parameter has protected factual gradients
                if 'factual_grad' in state and 'fisher_info' in state:
                    factual_grad = state['factual_grad']
                    fisher_info = state['fisher_info']
                    
                    # 1. Gradient Alignment Conflict Detection
                    # Compute cosine similarity between current grad and factual grad
                    # Add epsilon to prevent division by zero
                    dot_product = torch.sum(grad * factual_grad)
                    norm_grad = torch.norm(grad) + 1e-8
                    norm_factual = torch.norm(factual_grad) + 1e-8
                    cosine_sim = dot_product / (norm_grad * norm_factual)
                    
                    # If conflict detected (cosine_sim < threshold), apply Fisher penalty
                    if cosine_sim < self.conflict_threshold:
                        # 2. Fisher-Based Truth Importance Penalty
                        # The penalty pulls the parameter back towards its stable factual state
                        # scaled by the Fisher Information
                        if 'factual_param_state' in state:
                            factual_state = state['factual_param_state']
                            penalty = self.penalty_weight * fisher_info * (p - factual_state)
                            
                            # Add penalty to the gradient (equivalent to regularizing the loss)
                            grad.add_(penalty)
                
                # After adjusting the gradient, proceed with standard AdamW update
                # (The implementation below calls the standard AdamW step logic internally,
                # but since we modified `p.grad` in-place, the underlying C++ or Python 
                # AdamW implementation will use our conflict-adjusted gradients).
                
        # Call the parent class step to apply the modified gradients
        super().step()
        return loss

    def update_factual_memory(self, model, dataloader, device):
        """
        Computes the Fisher Information Matrix and Factual Gradients over a verified dataset.
        This acts as the 'consolidation phase' before learning new, potentially conflicting data.
        """
        model.train() # Must be in train mode to properly track gradients
        
        # Initialize storage
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        state = self.state[p]
                        # If state is empty, initialize AdamW keys first
                        if len(state) == 0:
                            state['step'] = torch.tensor(0.0)
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                        state['fisher_info'] = torch.zeros_like(p)
                        state['factual_grad'] = torch.zeros_like(p)
                        state['factual_param_state'] = p.clone().detach()

        total_samples = 0
        
        # Accumulate gradients and squared gradients (Fisher diagonal)
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            # Compute gradients
            model.zero_grad()
            loss.backward()
            
            # Accumulate
            with torch.no_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            self.state[p]['fisher_info'] += p.grad ** 2
                            self.state[p]['factual_grad'] += p.grad
            
            total_samples += 1

        # Average over the dataset
        with torch.no_grad():
            if total_samples > 0:
                for group in self.param_groups:
                    for p in group['params']:
                        if p.requires_grad:
                            self.state[p]['fisher_info'] /= total_samples
                            self.state[p]['factual_grad'] /= total_samples
