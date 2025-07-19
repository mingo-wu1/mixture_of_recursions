import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MoRConfig:
    """Configuration class for MoR models."""
    def __init__(self, model_name="mor-135m", **kwargs):
        self.model_name = model_name
        self.vocab_size = 49000
        self.max_seq_len = 2048
        self.num_recursion_steps = 3
        self.router_type = 'token_choice'  # 'token_choice' or 'expert_choice'
        self.sharing_strategy = 'cycle'  # 'cycle' or 'sequence'
        self.dropout = 0.1
        
        # Model size configurations based on paper
        if "135m" in model_name:
            self.d_model = 576
            self.num_layers = 30
            self.n_head = 9
            self.n_kv_head = 3  # GQA
            self.d_inter = 1536
        elif "360m" in model_name:
            self.d_model = 960
            self.num_layers = 32
            self.n_head = 15
            self.n_kv_head = 5  # GQA
            self.d_inter = 2560
        elif "730m" in model_name:
            self.d_model = 1536
            self.num_layers = 26
            self.n_head = 24
            self.n_kv_head = 8  # GQA
            self.d_inter = 4096
        elif "1.7b" in model_name:
            self.d_model = 2048
            self.num_layers = 24
            self.n_head = 32
            self.n_kv_head = 32  # Full attention
            self.d_inter = 8192
        
        self.d_head = self.d_model // self.n_head
        
        # Override with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class RMSNorm(nn.Module):
    """RMS Normalization as used in LLaMA."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        return cos_cached, sin_cached

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention as in paper."""
    def __init__(self, d_model, n_head, n_kv_head, d_head, max_seq_len=2048):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.d_head = d_head
        self.scale = 1 / math.sqrt(d_head)
        
        self.q_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_head * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_head * d_head, bias=False)
        self.out_proj = nn.Linear(n_head * d_head, d_model, bias=False)
        
        self.rotary_emb = RotaryEmbedding(d_head, max_seq_len)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_head, self.d_head).transpose(1, 2)
        
        # Apply rotary embedding
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand k, v for grouped query attention
        if self.n_kv_head != self.n_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        
        # Attention computation
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if attention_mask is None:
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
            attention_mask = attention_mask.bool()
        
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.out_proj(context)

class SiLUFeedForward(nn.Module):
    """SiLU-gated FFN as in LLaMA."""
    def __init__(self, d_model, d_inter):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_inter, bias=False)
        self.up_proj = nn.Linear(d_model, d_inter, bias=False)
        self.down_proj = nn.Linear(d_inter, d_model, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        gate = self.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class SharedBlock(nn.Module):
    """Shared Transformer block representing f(h; Φ')."""
    def __init__(self, d_model, n_head, n_kv_head, d_head, d_inter, max_seq_len=2048):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, n_head, n_kv_head, d_head, max_seq_len)
        self.ffn = SiLUFeedForward(d_model, d_inter)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x

class TokenChoiceRouter(nn.Module):
    """Token-Choice routing from paper Section 2.2.1."""
    def __init__(self, d_model, num_experts, router_arch='linear', z_loss_coef=1e-3, balance_loss_coef=1e-2):
        super().__init__()
        self.num_experts = num_experts
        
        if router_arch == 'linear':
            self.router = nn.Linear(d_model, num_experts, bias=False)
        elif router_arch == 'mlp':
            self.router = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, num_experts, bias=False)
            )
        
        self.z_loss_coef = z_loss_coef
        self.balance_loss_coef = balance_loss_coef

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, x.size(-1))
        
        # Router decision
        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)
        
        # Select expert (recursion depth) for each token
        selected_experts = torch.argmax(routing_probs, dim=-1)
        assigned_depths = selected_experts + 1  # 1-indexed
        
        # Calculate auxiliary losses
        aux_loss = self._compute_aux_loss(router_logits, routing_probs, selected_experts)
        
        return assigned_depths.view(batch_size, seq_len), aux_loss

    def _compute_aux_loss(self, logits, probs, experts):
        # Z-loss for numerical stability
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2)
        
        # Load balancing loss
        num_tokens = logits.size(0)
        expert_counts = torch.bincount(experts, minlength=self.num_experts).float()
        load_balance = expert_counts / num_tokens
        
        # Mean probability per expert
        mean_probs = probs.mean(dim=0)
        
        # Balance loss encourages uniform distribution
        balance_loss = self.num_experts * torch.sum(load_balance * mean_probs)
        
        return self.z_loss_coef * z_loss + self.balance_loss_coef * balance_loss

class ExpertChoiceRouter(nn.Module):
    """Expert-Choice routing from paper Section 2.2.1."""
    def __init__(self, d_model, capacity_factor=0.67, router_arch='linear', aux_loss_coef=1e-3):
        super().__init__()
        self.capacity_factor = capacity_factor
        
        if router_arch == 'linear':
            self.router = nn.Linear(d_model, 1, bias=False)
        elif router_arch == 'mlp':
            self.router = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.SiLU(),
                nn.Linear(d_model // 4, 1, bias=False)
            )
        
        self.aux_loss_coef = aux_loss_coef

    def forward(self, x, current_mask):
        # Only route tokens that are currently active
        active_tokens = x[current_mask]
        
        if active_tokens.numel() == 0:
            return current_mask, torch.tensor(0.0, device=x.device)
        
        # Compute routing scores
        scores = torch.sigmoid(self.router(active_tokens).squeeze(-1))
        
        # Select top-k tokens based on capacity
        num_to_keep = max(1, int(active_tokens.size(0) * self.capacity_factor))
        
        if num_to_keep >= active_tokens.size(0):
            return current_mask, torch.tensor(0.0, device=x.device)
        
        top_scores, top_indices = torch.topk(scores, num_to_keep)
        
        # Create new mask
        new_mask = torch.zeros_like(current_mask)
        active_positions = torch.where(current_mask)
        selected_positions = (active_positions[0][top_indices], active_positions[1][top_indices])
        new_mask[selected_positions] = True
        
        # Auxiliary loss encourages high scores for selected tokens
        aux_loss = -torch.mean(top_scores) * self.aux_loss_coef
        
        return new_mask, aux_loss

class MoRModel(nn.Module):
    """Mixture-of-Recursions Model."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        if config.sharing_strategy == 'middle_cycle':
            # Middle-Cycle: 首尾层独立，中间层共享
            self.first_layer = SharedBlock(config.d_model, config.n_head, config.n_kv_head, 
                                        config.d_head, config.d_inter, config.max_seq_len)
            self.last_layer = SharedBlock(config.d_model, config.n_head, config.n_kv_head,
                                        config.d_head, config.d_inter, config.max_seq_len)
            
            # 中间的共享层
            middle_layers = config.num_layers - 2  # 减去首尾层
            shared_middle_blocks = max(1, middle_layers // config.num_recursion_steps)
            
            self.shared_blocks = nn.ModuleList([
                SharedBlock(config.d_model, config.n_head, config.n_kv_head,
                        config.d_head, config.d_inter, config.max_seq_len) 
                for _ in range(shared_middle_blocks)
            ])
            
        else:
             # Shared blocks (parameter-efficient)
            blocks_per_recursion = max(1, config.num_layers // config.num_recursion_steps)
            self.shared_blocks = nn.ModuleList([
                SharedBlock(
                    config.d_model, 
                    config.n_head, 
                    config.n_kv_head,
                    config.d_head, 
                    config.d_inter,
                    config.max_seq_len
                ) for _ in range(blocks_per_recursion)
            ])
        
        # Router
        if config.router_type == 'token_choice':
            self.router = TokenChoiceRouter(config.d_model, config.num_recursion_steps)
        elif config.router_type == 'expert_choice':
            self.router = ExpertChoiceRouter(config.d_model)
        
        # Output
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_block_index(self, layer_idx):
        """Get shared block index based on sharing strategy."""
        num_blocks = len(self.shared_blocks)
        
        if self.config.sharing_strategy == 'cycle':
            return layer_idx % num_blocks
        elif self.config.sharing_strategy == 'sequence':
            layers_per_block = self.config.num_layers // num_blocks
            return min(layer_idx // layers_per_block, num_blocks - 1)
        else:
            raise ValueError(f"Unknown sharing strategy: {self.config.sharing_strategy}")

    def forward(self, input_ids, labels=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Input embeddings
        h = self.embedding(input_ids)
        
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        
        if self.config.router_type == 'token_choice':
            # Assign recursion depth to each token upfront
            assigned_depths, aux_loss = self.router(h)
            total_aux_loss += aux_loss
            
            # Process each layer
            for layer_idx in range(self.config.num_layers):
                # Only process tokens that need this layer
                needs_layer = assigned_depths * (self.config.num_layers // self.config.num_recursion_steps) > layer_idx
                
                if not needs_layer.any():
                    break
                
                # Get the shared block
                block_idx = self.get_block_index(layer_idx)
                block = self.shared_blocks[block_idx]
                
                # Apply to tokens that need processing
                h_processed = block(h, attention_mask=attention_mask)
                h = torch.where(needs_layer.unsqueeze(-1), h_processed, h)
        
        elif self.config.router_type == 'expert_choice':
            # Progressive filtering approach
            active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
            
            for layer_idx in range(self.config.num_layers):
                # Route to determine which tokens continue
                active_mask, aux_loss = self.router(h, active_mask)
                total_aux_loss += aux_loss
                
                if not active_mask.any():
                    break
                
                # Process active tokens
                block_idx = self.get_block_index(layer_idx)
                block = self.shared_blocks[block_idx]
                
                h_new = block(h, attention_mask=attention_mask)
                h = torch.where(active_mask.unsqueeze(-1), h_new, h)
        
        # Final output
        h = self.norm(h)
        logits = self.lm_head(h)
        
        loss = None
        if labels is not None:
            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Total loss includes auxiliary routing loss
            loss = lm_loss + total_aux_loss
        
        return logits, loss

    def count_parameters(self):
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
