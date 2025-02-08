"""
Hypothetical training script for the DeepSeek model.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import Transformer, ModelArgs


# --- Configuration --- (No changes here)
out_dir = 'out-deepseek'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'deepseek-train'
wandb_run_name = 'deepseek'
dataset = 'shakespeare'
gradient_accumulation_steps = 8
batch_size = 4
block_size = 4096
n_layers = 27
n_dense_layers = 1
n_heads = 16
n_routed_experts = 64
n_shared_experts = 2
n_activated_experts = 6
n_expert_groups = 1
n_limited_groups = 1
score_func = 'softmax'
route_scale = 1.0
q_lora_rank = 0
kv_lora_rank=512
qk_nope_head_dim=128
qk_rope_head_dim=64
v_head_dim=128
original_seq_len=4096
rope_theta = 10000.0
rope_factor = 40
beta_fast = 32
beta_slow = 1
mscale = 1.
dim = 2048
inter_dim = 10944
moe_inter_dim = 1408
vocab_size = 102400
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16'
compile = True


# --- Helper Functions ---
def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # NO dtype here.  Keep x and y as int64
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        # NO dtype here. Keep x and y as int64
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, ctx, eval_iters, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            with ctx:
                logits = model(X)
                logits = logits.view(-1, logits.size(-1))
                Y = Y.view(-1)
                loss = torch.nn.functional.cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# --- Main Training Script ---

if __name__ == '__main__':
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # --- Data Loading ---
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.int64, mode='r')  # Load int64
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.int64, mode='r')    # Load int64

    # --- Model Initialization ---
    if init_from == 'scratch':
        model_args = dict(
            n_layers=n_layers,
            n_dense_layers=n_dense_layers,
            n_heads=n_heads,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            n_activated_experts=n_activated_experts,
            n_expert_groups = n_expert_groups,
            n_limited_groups = n_limited_groups,
            score_func=score_func,
            route_scale = route_scale,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            original_seq_len = original_seq_len,
            rope_theta = rope_theta,
            rope_factor = rope_factor,
            beta_fast=beta_fast,
            beta_slow = beta_slow,
            mscale=mscale,
            max_batch_size=batch_size,
            max_seq_len=block_size,
            vocab_size=vocab_size,
            dim = dim,
            inter_dim = inter_dim,
            moe_inter_dim = moe_inter_dim,
            dtype=dtype,
        )

        model_args = ModelArgs(**model_args)
        model = Transformer(model_args)
    elif init_from == 'resume':
       ckpt_path = os.path.join(out_dir, 'ckpt.pt')
       checkpoint = torch.load(ckpt_path, map_location=device)
       model_args = checkpoint['model_args']
       model = Transformer(model_args)
       state_dict = checkpoint['model']
       unwanted_prefix = '_orig_mod.'
       for k, v in list(state_dict.items()):
          if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
       model.load_state_dict(state_dict)
       iter_num = checkpoint.get('iter_num', 0)
       best_val_loss = checkpoint.get('best_val_loss', 1e9)

    model.to(device)

    # --- Optimizer and Scaler ---
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    if init_from == 'resume':
       optimizer.load_state_dict(checkpoint['optimizer'])
    if compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # --- Training Loop ---
    iter_num = 0
    best_val_loss = 1e9
    X, Y = get_batch('train', train_data, val_data)  # Initial batch
    t0 = time.time()
    local_iter_num = 0

    while True:
        lr = get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, eval_iters, train_data, val_data)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': {},
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num == 0 and eval_only:
            break

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits = model(X)
                logits = logits.view(-1, logits.size(-1))
                Y_flat = Y.view(-1)
                loss = torch.nn.functional.cross_entropy(logits, Y_flat)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train', train_data, val_data)  # Get a NEW batch *after* forward pass
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()
