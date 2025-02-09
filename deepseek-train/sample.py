import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig  # Import AutoConfig
from model import Transformer, ModelArgs
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
def generate(model: Transformer, prompt_tokens: List[List[int]], max_new_tokens: int, eos_id: int, temperature: float = 1.0, top_k: Optional[int] = None) -> List[List[int]]:
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
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file (10m.json)")  # Now expects 10m.json
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

     # --- Load Model Args (from 10m.json) ---
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
            model_args = ModelArgs(**config['model_args']) # Get args from the 'model_args' dictionary
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: {args.config} is not a valid JSON file or is missing 'model_args': {e}")
        exit(1)


    # --- Load Tokenizer ---
    # Check if the tokenizer files exist.  This is *critical*.
    tokenizer_config_path = os.path.join(args.ckpt_path, "tokenizer_config.json")
    tokenizer_path = os.path.join(args.ckpt_path, "tokenizer.json")
    if not (os.path.exists(tokenizer_config_path) and os.path.exists(tokenizer_path)):
        print(f"CRITICAL ERROR: Tokenizer files (tokenizer_config.json and tokenizer.json) not found in {args.ckpt_path}.")
        print("You MUST have the correct tokenizer files to generate text.  These files should have been")
        print("copied to the checkpoint directory by the convert.py script, or be present from the original")
        print("source of the model.  Without the tokenizer, the model cannot function.")
        exit(1)


    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=True)
    # --- Create Model ---
    #model = AutoModelForCausalLM.from_config(config, trust_remote_code=True) # This does not load the weights!
    print(f"bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token_id: {tokenizer.eos_token_id}")
    print(f"pad_token_id: {tokenizer.pad_token_id}")
    print(f"unk_token_id: {tokenizer.unk_token_id}")

    model = Transformer(model_args).to(device) #use our transfomer

    # --- Load Model Weights (Handles Sharded and Merged Checkpoints) ---

    if world_size > 1:
       # Load the sharded checkpoint for the current process
       model_file = os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
       print(model_file)
       try:
          state_dict = torch.load(model_file, map_location=device)
          model.load_state_dict(state_dict, strict=False) #use strict=False since model expect scale but is not there
       except FileNotFoundError:
           print(f"Error: Sharded model file not found: {model_file}")
           print("       Make sure you are using the correct --ckpt-path and that you have trained")
           print("       with model parallelism (multiple GPUs). If you trained on a single GPU,")
           print("       the checkpoint might be a single file (ckpt.pt).")
           exit(1)
    #try to load ckpt.pt
    elif os.path.exists(os.path.join(args.ckpt_path, "ckpt.pt")):
        print("Loading merged checkpoint from ckpt.pt")
        checkpoint = torch.load(os.path.join(args.ckpt_path, "ckpt.pt"), map_location=device)
        if 'model' in checkpoint:
          model.load_state_dict(checkpoint['model'])
        else:
          model.load_state_dict(checkpoint) #, strict=False)
    else:
        print(f"Error: No checkpoint file found in {args.ckpt_path}.  Expected either sharded files")
        print(f"       (e.g., model0-mp4.safetensors) or a single ckpt.pt file.")
        exit(1)

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
