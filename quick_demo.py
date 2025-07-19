import torch
from mor_model import MoRModel, MoRConfig

def demo_mor():
    """Quick demo of MoR model."""
    print("MoR (Mixture-of-Recursions) Demo")
    print("="*50)
    
    # Test different configurations
    configs = [
        ("MoR-135M Token-Choice", MoRConfig("mor-135m", router_type='token_choice')),
        ("MoR-135M Expert-Choice", MoRConfig("mor-135m", router_type='expert_choice')),
        ("MoR-360M Token-Choice", MoRConfig("mor-360m", router_type='token_choice')),
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        print(f"  D_model: {config.d_model}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Recursions: {config.num_recursion_steps}")
        print(f"  Router: {config.router_type}")
        
        model = MoRModel(config)
        params = model.count_parameters()
        
        print(f"  Parameters: {params['total']:,}")
        print(f"  Expected reduction: ~{config.num_recursion_steps}x vs vanilla")
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 10))  # Small vocab for demo
        config.vocab_size = 1000  # Adjust for demo
        
        with torch.no_grad():
            logits, loss = model(input_ids)
            print(f"  Output shape: {logits.shape}")
            print(f"  Memory efficient: {'✓' if params['total'] < 50_000_000 else '✗'}")

if __name__ == "__main__":
    demo_mor()
