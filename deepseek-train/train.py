import os
import sys
import time
import math
import json  # Import the json module
from contextlib import nullcontext
import shutil
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# --- Import model and ModelArgs ---
from model import Transformer, ModelArgs
# No need for importlib, we are going to use the config.json file.
#import importlib  # Used for loading config #REMOVED
from transformers import AutoConfig #we load a config to get the model arguments

# --- Configuration ---
out_dir = 'out-deepseek'
eval_interval = 2000
log_interval = 1  # Log more frequently during debugging
eval_iters = 20  # Reduce for faster debugging
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch', 'resume'

# --- WandB ---
wandb_log = False  # Turn on to use wandb
wandb_project = 'deepseek-train'
wandb_run_name = 'run'

# --- Data ---
dataset = 'shakespeare' # or the name of your dataset
gradient_accumulation_steps = 32  # Adjust based on your GPU and batch_size
batch_size = 4 # Reduced batch size, adjust this value if you have a better GPU
block_size = 128 # Reduced block size, adjust.
# block_size = 4096  # original value

# --- Optimizer ---
learning_rate = 6e-4
#max_iters = 600000
max_iters = 400       # Limit to 400 iterations for testing
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 200   # Proportional to max_iters
lr_decay_iters = 400 # Proportional to max_iters
min_lr = 6e-5

# --- DDP ---
backend = 'nccl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # Use bfloat16 if available
compile = False # Disable compile by default.


# --- Helper Functions ---
def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
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

    # --- Setup DDP ---
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


   # --- Load Configuration from Command Line ---
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_file>") #e.g. configs/10m.json
        sys.exit(1)
    config_file = sys.argv[1]  # Get config file path from command line, should be 10m.json

    # --- Load Model Args (from 10m.json) ---
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            model_args = ModelArgs(**config['model_args']) # Get args from the 'model_args' dictionary
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_file}")
        exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: {config_file} is not a valid JSON file or is missing 'model_args': {e}")
        exit(1)

    # Create a config dictionary in the format that Hugging Face expects.  This is
    # what will get saved as config.json.  We *combine* information from
    # your 10m.json *and* from the example config.json you provided.

    config_dict = {
        "architectures": [
          "DeepseekV3ForCausalLM"
        ],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "auto_map": {
          "AutoConfig": "configuration_deepseek.DeepseekV3Config",
          "AutoModel": "modeling_deepseek.DeepseekV3Model",  #<-- You will need to create this.
          "AutoModelForCausalLM": "modeling_deepseek.DeepseekV3ForCausalLM" #<--- and this.
        },
        # All your model parameters from 10m.json:
        **config['model_args'],  # Add all the parameters from model_args
        "model_type": "deepseek_v3",  # VERY IMPORTANT: This must match your config class name.
    }


    # Save the *complete* configuration as config.json
    config_json_path = os.path.join(out_dir, 'config.json')
    if master_process:  # Only save from the master process
       with open(config_json_path, 'w') as f:
           json.dump(config_dict, f, indent=4) #save config dict and not model_args_dict
       print(f"Saved Hugging Face compatible config to {config_json_path}")

        # Copy the configuration_deepseek.py file to the output directory
       try:
           shutil.copy("configs/configuration_deepseek.py", out_dir) #copy file
           print("configuration_deepseek.py copied to checkpoint directory.")
       except FileNotFoundError:
            print("Error: configuration_deepseek.py not found.  Make sure it exists.")
            exit(1)


    # --- Model Initialization ---
    if init_from == 'scratch':
        model = Transformer(model_args)
        print(f"Parameter count: {model.get_parameter_count():,}")
    elif init_from == 'resume':
       ckpt_path = os.path.join(out_dir, 'ckpt.pt')
       checkpoint = torch.load(ckpt_path, map_location=device)
       model_args = checkpoint['model_args']  # Load saved ModelArgs
       model = Transformer(model_args)      # Recreate the model
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
       optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer state
    #if compile:  # DISABLE COMPILATION FOR NOW
    #    model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # --- Training Loop ---
    iter_num = 0
    best_val_loss = 1e9
    t0 = time.time()
    local_iter_num = 0

    while True:
        lr = get_lr(iter_num, learning_rate, warmup_iters, lr_decay_iters, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, eval_iters, train_data, val_data)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args.__dict__,  # Save ModelArgs as a dict
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config_dict, #save the config dict
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num == 0 and eval_only:
            break

        accumulated_loss = None

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

            X, Y = get_batch('train', train_data, val_data)

            with ctx:
                logits = model(X)
                logits = logits.view(-1, logits.size(-1))
                Y_flat = Y.view(-1)
                logits = logits.to(torch.float32)  # Cast logits to float32
                loss = torch.nn.functional.cross_entropy(logits, Y_flat)
                loss = loss / gradient_accumulation_steps  # scale the loss *before* accumulating
            scaled_loss = scaler.scale(loss) # Scale
            if accumulated_loss is None:
                accumulated_loss = scaled_loss
            else:
                accumulated_loss += scaled_loss

        # --- BACKWARD PASS OUTSIDE THE INNER LOOP---
        accumulated_loss.backward()

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
            lossf = loss.item() * gradient_accumulation_steps  # this is now correct
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()
