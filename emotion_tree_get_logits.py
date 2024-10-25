# generate and save logits for 5000 prompts, using GPT2 and Llama 3.1 7B/80B/405B

from nnsight import CONFIG
from nnsight import LanguageModel
import os
import numpy as np
import pickle
import torch
import more_itertools



prompt_name = 'emotional_sentence_chatgpt4_5000'
emotion_prompt = " The emotion in this sentence is"


device = torch.device('mps')
with open('data/{}.txt'.format(prompt_name), 'r') as file:
    prompts = [line.strip() + emotion_prompt for line in file]


def get_logits(model_name, filename, use_remote=False, batch_size=128):

    cache_dir = f'cache/hidden_states_{filename}/'
    os.makedirs(cache_dir, exist_ok=True)

    # print(model_name)
    model = LanguageModel(model_name, device_map="auto")
    """
    logits_list0 = []
    for i in range(0, len(prompts)):
        if i % 500 == 0:
            print(i)

        with model.trace(prompts[i], remote=use_remote) as runner:
            logits = model.lm_head.output.save()
        logits = logits[:, -1, :]
        logits = torch.nn.functional.softmax(logits, dim=-1)#.detach().cpu().to(torch.float32).numpy()
        logits_list0.append(logits)
    with open('cache/hidden_states_{}/{}_logits_list.pkl'.format(filename, prompt_name), 'wb') as f:
        pickle.dump(logits_list0, f)
    """

    logits_list = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        if i % (10 * batch_size) == 0:
            print(f"Processing batch {i // batch_size}...")
        with model.trace(batch_prompts, remote=use_remote) as runner:
            logits_batch = model.lm_head.output.save()
        logits_batch = logits_batch[:, -1, :]
        logits_batch = torch.nn.functional.softmax(logits_batch, dim=-1).detach().cpu()
        logits_list.append(logits_batch)

    logits = torch.cat(logits_list)
    with open(f'cache/hidden_states_{filename}/{prompt_name}_logits_list.pt', 'wb') as f:
        torch.save(logits, f)


# GPT 2
#get_logits('gpt2', 'gpt2', False)

# GPT 6b
#get_logits('EleutherAI/gpt-j-6b', 'gpt-j-6b', True)

# Llama 8B
#get_logits('meta-llama/Meta-Llama-3.1-8B', 'llama-8', True)

# Llama 70B
get_logits('meta-llama/Meta-Llama-3.1-70B', 'llama-70', True)

# Llama 405B
#get_logits('meta-llama/Meta-Llama-3.1-405B', 'llama-405', True)

