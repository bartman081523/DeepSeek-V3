import os
import json
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from model import Transformer, ModelArgs
import importlib
from argparse import ArgumentParser
from typing import List, Optional  # Import Optional


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    """Samples a token from the logits using temperature scaling and top-k filtering."""
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')  # Use -inf for masking
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.inference_mode()
def generate(model: Transformer, prompt_tokens: List[List[int]], max_new_tokens: int, eos_id: int, temperature: float = 1.0, top_k: Optional[int] = None) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.
        top_k (Optional[int], optional): The top-k value for sampling. Defaults to None.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = sample(logits[:, -1, :], temperature, top_k) #sample from last token
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens



def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file (e.g., configs/10m.py)")
    parser.add_argument("--prompt", type=str, default="", help="Initial prompt for text generation")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of tokens to generate")
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
    if 'RANK' in os.environ:  # Check if running in a distributed environment
        dist.init_process_group(backend="nccl")  # Or 'gloo' if NCCL isn't available
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}' # each process on their own cuda
    else:
        rank = 0
        world_size = 1


    # --- Load Model Args (from config file) ---
    config_name = os.path.splitext(os.path.basename(args.config))[0]  # Extract config name
    config_module = importlib.import_module(f"configs.{config_name}")
    model_args = config_module.model_args  # Access ModelArgs

    # --- Load Tokenizer (CRUCIAL: Use AutoTokenizer with the checkpoint directory) ---
    # The tokenizer MUST be loaded from the same directory as the checkpoint.
    # convert.py *should* have copied the tokenizer files there.
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)

    # --- Create/Load Model ---
    model = Transformer(model_args).to(device)

    # --- Load Model Weights (Corrected for distributed setup) ---
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

    model.eval()  # Set the model to evaluation mode


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
            completion_tokens = generate(model, [prompt_tokens], args.max_new_tokens, tokenizer.eos_token_id, args.temperature, args.top_k)
            completion = tokenizer.decode(completion_tokens[0]) #, skip_special_tokens=True) Remove skip special tokens, it may cause problems
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
