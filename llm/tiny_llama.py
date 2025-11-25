# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
import transformers
from transformers import pipeline, Pipeline

pipe: Pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

pipe.model.config._attn_implementation = "eager"
print(type(pipe), type(pipe.model))
transformers.pipelines.text_generation.TextGenerationPipeline
transformers.models.llama.modeling_llama.LlamaForCausalLM
print(pipe.model.config._attn_implementation)

model: transformers.models.llama.modeling_llama.LlamaForCausalLM = pipe.model
model.eval()
print(model)

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(type(prompt), prompt)
ep = torch.export.export(model, (prompt,))

outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...
