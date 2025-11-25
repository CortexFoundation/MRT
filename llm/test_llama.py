import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

print(inputs)
import torch
transformers.cache_utils.DynamicCache
ep = torch.export.export(model, (inputs.input_ids,))

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)
