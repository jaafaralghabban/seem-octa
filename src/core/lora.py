"""
Low-Rank Adaptation (LoRA) Implementation for SEEM Model

This module provides LoRA adapters for parameter-efficient fine-tuning of the
SEEM foundation model. Only ~1.25% of parameters are trainable with LoRA.

References:
    - LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation module.
    
    Implements the low-rank decomposition: W' = W + BA where B ∈ R^{d×r} and A ∈ R^{r×k}
    with r << min(d, k).
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        r: Rank of the low-rank matrices (default: 16)
        alpha: Scaling factor (default: 32)
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 16, 
        alpha: int = 32
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Initialize A with Kaiming, B with zeros (so initial output is zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation: scaling * B(A(x))"""
        return self.lora_B(self.lora_A(x)) * self.scaling
    
    def __repr__(self) -> str:
        return f"LoRAAdapter(r={self.r}, alpha={self.alpha}, scaling={self.scaling:.2f})"


class LoRAInjectedLinear(nn.Module):
    """
    Linear layer with injected LoRA adapter.
    
    Wraps an existing Linear layer and adds a parallel LoRA path.
    Output = original_linear(x) + lora_adapter(x)
    
    Args:
        original_layer: The original nn.Linear layer to wrap
        r: LoRA rank (default: 16)
        alpha: LoRA scaling factor (default: 32)
    """
    
    def __init__(
        self, 
        original_layer: nn.Linear, 
        r: int = 16, 
        alpha: int = 32
    ):
        super().__init__()
        self.original_layer = original_layer
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        device = original_layer.weight.device
        
        self.lora_adapter = LoRAAdapter(
            in_features, out_features, r=r, alpha=alpha
        ).to(device)
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining original layer and LoRA adapter."""
        return self.original_layer(x) + self.lora_adapter(x)
    
    @property
    def weight(self) -> torch.Tensor:
        """Return original weight for compatibility."""
        return self.original_layer.weight
    
    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Return original bias for compatibility."""
        return self.original_layer.bias


def inject_lora(
    module: nn.Module, 
    r: int = 16, 
    alpha: int = 32,
    target_modules: Optional[list] = None
) -> int:
    """
    Recursively inject LoRA adapters into Linear layers.
    
    Args:
        module: PyTorch module to inject LoRA into
        r: LoRA rank
        alpha: LoRA scaling factor
        target_modules: Optional list of module name patterns to target
        
    Returns:
        Number of layers injected
    """
    injected_count = 0
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Only inject if input features > rank (makes sense mathematically)
            if child.in_features > r:
                # Check if we should target this module
                if target_modules is None or any(t in name for t in target_modules):
                    setattr(module, name, LoRAInjectedLinear(child, r=r, alpha=alpha))
                    injected_count += 1
        elif len(list(child.children())) > 0:
            # Recurse into child modules
            injected_count += inject_lora(child, r=r, alpha=alpha, target_modules=target_modules)
    
    return injected_count


def merge_lora_weights(model: nn.Module, device: str = 'cuda') -> nn.Module:
    """
    Merge LoRA weights into the base model for efficient inference.
    
    This creates a new model where LoRA adapters are merged into the original
    weights: W_merged = W_original + scaling * B @ A
    
    Args:
        model: Model with LoRA adapters
        device: Target device for merged model
        
    Returns:
        New model with merged weights (no LoRA adapters)
    """
    merged_model = copy.deepcopy(model)
    
    for name, module in merged_model.named_modules():
        if isinstance(module, LoRAInjectedLinear):
            # Get original weights
            original_weight = module.original_layer.weight.data
            
            # Get LoRA weights
            lora_A_weight = module.lora_adapter.lora_A.weight.data
            lora_B_weight = module.lora_adapter.lora_B.weight.data
            scaling = module.lora_adapter.scaling
            
            # Merge: W' = W + scaling * B @ A
            merged_weight = original_weight + (lora_B_weight @ lora_A_weight) * scaling
            
            # Create new Linear layer
            in_features = module.original_layer.in_features
            out_features = module.original_layer.out_features
            has_bias = module.original_layer.bias is not None
            
            new_layer = nn.Linear(in_features, out_features, bias=has_bias).to(device)
            new_layer.weight.data = merged_weight
            
            if has_bias:
                new_layer.bias.data = module.original_layer.bias.data
            
            # Replace in model
            parent_name_parts = name.split('.')[:-1]
            if parent_name_parts:
                parent_module = merged_model.get_submodule('.'.join(parent_name_parts))
            else:
                parent_module = merged_model
            
            layer_name = name.split('.')[-1]
            setattr(parent_module, layer_name, new_layer)
    
    return merged_model


def save_merged_model(model: nn.Module, save_path: str, device: str = 'cuda') -> None:
    """
    Merge LoRA weights and save the model for deployment.
    
    Args:
        model: Model with LoRA adapters
        save_path: Path to save the merged model
        device: Target device
    """
    print("\nMerging LoRA weights into base model...")
    merged_model = merge_lora_weights(model, device)
    
    print(f"Saving merged model to: {save_path}")
    torch.save({'model': merged_model.state_dict()}, save_path)
    print("✓ Merged model saved successfully")


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """
    Count total and trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params, trainable_percentage)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    
    return total_params, trainable_params, percentage


def freeze_base_model(model: nn.Module) -> None:
    """
    Freeze all parameters except LoRA adapters and decoder.
    
    Args:
        model: SEEM model with LoRA adapters
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA adapters and decoder
    for name, param in model.named_parameters():
        if 'lora_adapter' in name or 'sem_seg_head' in name:
            param.requires_grad = True


def get_lora_state_dict(model: nn.Module) -> Dict[str, Any]:
    """
    Extract only LoRA parameters from model state dict.
    
    Useful for saving/loading only the adapter weights.
    
    Args:
        model: Model with LoRA adapters
        
    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_adapter' in name:
            lora_state_dict[name] = param.data
    return lora_state_dict
