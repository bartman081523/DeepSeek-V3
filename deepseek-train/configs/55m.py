from model import ModelArgs

model_args = ModelArgs(
    n_layers=2,
    n_dense_layers=2,
    n_heads=8,
    dim=512, # Modified
    inter_dim=4*512,
    vocab_size=102400,  # Your vocab size
    max_seq_len=4096,
    qk_nope_head_dim=64, #modified
    qk_rope_head_dim=64, #modified
    v_head_dim=64, #modified
    n_routed_experts=0,
    n_shared_experts=0,
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
