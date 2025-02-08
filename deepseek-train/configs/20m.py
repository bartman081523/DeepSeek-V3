
from model import ModelArgs

model_args = ModelArgs(
    n_layers=2,
    n_dense_layers=2,
    n_heads=4,
    dim=288,  # Adjusted
    inter_dim=4*288,
    vocab_size=102400, #YOUR VOCAB SIZE
    max_seq_len=1024,  # Same as block_size
    qk_nope_head_dim=32,
    qk_rope_head_dim=32,
    v_head_dim=32,
    n_routed_experts=0, # MoE disabled
    n_shared_experts=0,   # MoE disabled
    n_activated_experts=0,
    n_expert_groups = 1,
    n_limited_groups = 1,
    score_func = "softmax",
    route_scale = 1.0,
    q_lora_rank = 0,
    kv_lora_rank = 0,
    original_seq_len = 4096,
    rope_theta = 10000.0,
    rope_factor = 1.0,
    beta_fast = 32,
    beta_slow = 1,
    mscale = 1.0,
    dtype="bfloat16"

)
