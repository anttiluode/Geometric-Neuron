import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import sys

# Import your custom architecture from the training script
try:
    from moire_attention_gpt2 import MoireGPT, MoireGPTConfig
except ImportError:
    print("Error: Could not import MoireGPT. Make sure moire_attention_gpt2.py is in the same folder.")
    sys.exit(1)

def load_moire_model(weights_path='moire_gpt_weights.pt', device='cuda'):
    print(f"Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print(f"Initializing Moiré wave-field architecture...")
    # These MUST match the exact configuration used during training
    config = MoireGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=8,
        n_embd=256,
        max_seq_len=129,
        gamma_slots=8,
        use_theta_gating=True
    )

    model = MoireGPT(config)
    
    print(f"Loading trained weights from {weights_path}...")
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Error: Could not find {weights_path}. Did the training finish successfully?")
        sys.exit(1)
        
    model.to(device)
    model.eval() # Set to evaluation mode (turns off dropout)
    
    return model, tokenizer, config

def generate_text(model, tokenizer, config, prompt, max_new_tokens=60, temperature=0.8, top_k=40, device='cuda'):
    # Convert text prompt to token IDs
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print("Moiré: ", end="", flush=True)
    
    # Generate tokens one by one
    for _ in range(max_new_tokens):
        # Crop context to the max sequence length the model can handle
        idx_cond = input_ids[:, -(config.max_seq_len - 1):]
        
        with torch.no_grad():
            # Calculate Re[Q·conj(K)] interference for the next token
            logits, _ = model(idx_cond)
            
        # Get the logits for the last token only, apply temperature
        logits = logits[:, -1, :] / temperature
        
        # Top-K filtering (prevents the model from picking wildly wrong words)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to the running sequence
        input_ids = torch.cat((input_ids, next_token), dim=1)
        
        # Decode and print just the new token for a streaming effect
        word = tokenizer.decode(next_token[0].tolist())
        print(word, end="", flush=True)
        
    print() # Newline when done
    return input_ids

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"=== Moiré Attention LLM Interface ===")
    print(f"Running on: {device.upper()}")
    
    model, tokenizer, config = load_moire_model(device=device)
    
    print("\n" + "="*50)
    print("Moiré field is stable and ready.")
    print("Note: This is a 16M parameter base model trained on WikiText.")
    print("It completes your sentences. It is not an instruction-following chatbot.")
    print("Type 'quit' or 'exit' to close the connection.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Closing Moiré field connection...")
                break
            if not user_input.strip():
                continue
                
            # Run the wave-interference generation
            generate_text(model, tokenizer, config, user_input, device=device)
            
        except KeyboardInterrupt:
            print("\nClosing Moiré field connection...")
            break

if __name__ == "__main__":
    main()