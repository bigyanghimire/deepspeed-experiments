import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)

model.eval()

# Store activation stats
activation_stats = defaultdict(list)

def hook_fn(name):
    def hook(module, inputs, output):
        x = inputs[0]   # input activation to linear layer
        # Mean abs activation per hidden channel
        mean_abs = x.abs().mean(dim=(0, 1)).detach().cpu()
        activation_stats[name].append(mean_abs)
    return hook

# Register hooks on all linear layers
# hooks = []
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         hooks.append(module.register_forward_hook(hook_fn(name)))

# Sample calibration data
texts = [
    "The theory of relativity changed modern physics.",
    "Large language models are powerful for reasoning tasks.",
    "Quantization reduces memory usage and improves inference speed.",
    "Deep learning models contain activation outliers."
]
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:100]")
texts = dataset["text"][:100]
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output=model(**inputs)
        logits = output.logits
        
        # Get the highest probability token ID for each position
        predicted_token_ids = torch.argmax(logits, dim=-1)
        
        # Decode the sequence
        decoded_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        print("Predicted text from logits:", decoded_text)

# Analyze results
# for layer_name, acts in activation_stats.items():
#     acts = torch.stack(acts).mean(dim=0)

#     num_channels = acts.numel()
#     top_k = max(1, int(0.01 * num_channels))

#     top_vals, top_idx = torch.topk(acts, top_k)

#     print(f"\nLayer: {layer_name}")
#     print(f"Channels: {num_channels}")
#     print(f"Top 1% channels: {top_k}")
#     print(f"Mean activation (all): {acts.mean():.4f}")
#     print(f"Mean activation (top 1%): {top_vals.mean():.4f}")
#     print(f"Ratio: {(top_vals.mean() / acts.mean()):.2f}x")