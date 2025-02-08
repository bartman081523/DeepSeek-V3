from model import ModelArgs

model_args = ModelArgs(
    n_layers=6, # Number of layers
    n_dense_layers=6,
    n_heads=8,   # Number of heads
    dim=512,    # Embedding dimension
    inter_dim=4*512,   # Intermediate dimension
    vocab_size=102400, # Your vocab_size
    max_seq_len=4096,  # maximum context length
    qk_nope_head_dim=64,
    qk_rope_head_dim=64,
    v_head_dim=64,
    n_routed_experts=0,  # MoE disabled
    n_shared_experts=0,   # MoE disabled
    n_activated_experts=0,  # MoE disabled
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
