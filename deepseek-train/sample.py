import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from model import Transformer, ModelArgs  # Keep this for now
from argparse import ArgumentParser
from typing import List, Optional
import json


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    """Samples a token from the logits using temperature scaling and top-k filtering."""
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')  # Use -inf for masking
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate(model, prompt_tokens, max_new_tokens, eos_id, temperature=1.0, top_k=None):
    """Generates new tokens, adapted from original generate.py."""
    prompt_len = len(prompt_tokens)
    if prompt_len > model.max_seq_len:
        prompt_tokens = prompt_tokens[-model.max_seq_len:]  # Truncate prompt if needed
    total_len = min(model.max_seq_len, max_new_tokens + prompt_len)
    tokens = torch.full((1, total_len), eos_id, dtype=torch.long, device=model.device)  # Initialize with eos_token_id
    tokens[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device=model.device)
    prev_pos = 0
    for cur_pos in range(prompt_len, total_len):
        logits = model(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = sample(logits[:, -1, :], temperature, top_k)
        tokens[:, cur_pos] = next_token
        if next_token == eos_id:
            break
        prev_pos = cur_pos
    return tokens[0, prompt_len:cur_pos + 1].tolist()


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint directory")
    # No need for --config, it's in ckpt-path now
    parser.add_argument("--prompt", type=str, default="", help="Initial prompt for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling parameter")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    # --- Setup: Seeding, Device, and Distributed ---
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(args.seed)  # For multi-GPU
    else:
        device = 'cpu'
        print("No CUDA, using CPU")

    # Handle distributed setup (if applicable)
    if 'RANK' in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        rank = 0
        world_size = 1

    # --- Load Configuration and Tokenizer with trust_remote_code ---
    config = AutoConfig.from_pretrained(args.ckpt_path, trust_remote_code=True)  # Load config
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)


    # --- Create Model (using AutoModelForCausalLM) ---
    # Now that the files are set, we should create a new class DeepseekV3ForCausalLM.
    # This way the loading could be done with AutoModelForCausalLM instead of calling our implementation
    #model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = Transformer(config).to(device) #keep our Transformer

    # --- Load Model Weights ---
    if world_size > 1:
       # Load the sharded checkpoint for the current process
       model_file = os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
       state_dict = torch.load(model_file, map_location=device)
       model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(args.ckpt_path) and os.path.isdir(args.ckpt_path):
        # Load the merged checkpoint if it's a directory
        model_file = os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict, strict=False) #use strict=False since model expect scale but is not there
    else:
       checkpoint = torch.load(args.ckpt_path, map_location=device)
       if 'model' in checkpoint:
          model.load_state_dict(checkpoint['model'])  # Load from the 'model' key
       else:
          model.load_state_dict(checkpoint)

    model.eval()  # Ensure the model is in evaluation mode

   # --- Prepare Prompt ---
    if args.interactive:
      messages = []
      while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, prompt_tokens, args.max_new_tokens, tokenizer.eos_token_id, args.temperature, args.top_k)
            completion = tokenizer.decode(completion_tokens) #, skip_special_tokens=True) Remove skip special tokens, it may cause problems
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
         # --- Prepare Prompt ---
        if args.prompt == "": # if prompt is empty, use the end of text token.
           start_ids = [tokenizer.eos_token_id]
        else:
           start_ids = tokenizer.encode(args.prompt) # Encode the prompt

        x = torch.tensor([start_ids], dtype=torch.long, device=device)

        # --- Generate ---
        with torch.no_grad():
            y = model.generate(x, args.max_new_tokens, args.temperature, args.top_k)
            print(tokenizer.decode(y[0].tolist())) #before this was missing
            print("---------------")

if __name__ == "__main__":
    main()
