import torch
from awq.auto import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

quant_path = '/mnt/140/llama3/Meta-Llama-3-8B-Instruct-hf-AWQ'

# Load model
model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)

tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "You're standing on the surface of the Earth. "\
        'You walk one mile south, one mile west and one mile north. '\
        'You end up exactly where you started. Where are you?'

chat = [
    {
        'role': 'system',
        'content': 'You are a concise assistant that helps answer questions.'
    },
    {
        'role': 'user',
        'content': prompt
    },
]

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids('<|eot_id|>')
]

tokens = tokenizer.apply_chat_template(chat, return_tensors='pt')
tokens = tokens.to(torch.device('cuda'))

# Generate output
generation_output = model.generate(tokens,
                                   streamer=streamer,
                                   max_new_tokens=64,
                                   eos_token_id=terminators)

res = tokenizer.decode(generation_output[0].cpu().numpy().tolist())
print(f'token_ids: {generation_output}')
print(f'output: {res}')
