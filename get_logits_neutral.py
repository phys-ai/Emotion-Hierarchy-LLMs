from nnsight import CONFIG
from nnsight import LanguageModel
import os
import numpy as np
import pickle
import torch
import more_itertools



prompt_name = 'chatgpt4o_scenario_neutral'
emotion_prompt = "\nAs a person with Autism Spectrum Disorder, I think the emotion involved in this situation is"

device = torch.device('mps')
with open('data/{}.txt'.format(prompt_name), 'r') as file:
    prompts = [line.strip() + emotion_prompt for line in file]
print('Number of prompts:', len(prompts))
print(prompts[0])

# print(model_name)
model = LanguageModel('meta-llama/Meta-Llama-3.1-405B', device_map="auto")
# print(model)

batch_size = 128
logits_list = []
hidden_states_list = []
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i + batch_size]
    if i % (10 * batch_size) == 0:
        print(f"Processing batch {i // batch_size}...")
    with model.trace(batch_prompts, remote=True) as runner:
        logits_batch = model.lm_head.output.save()
        hidden_states_batch = [model.model.layers[-1].output[0].save() for ilayer in [-50,-20,-10,-2,-1]]
    logits_batch = logits_batch[:, -1, :]
    logits_batch = torch.nn.functional.softmax(logits_batch, dim=-1).detach().cpu()
    logits_list.append(logits_batch)

    hidden_states_batch = torch.stack(hidden_states_batch) 
    hidden_states_list.append(hidden_states_batch)

logits = torch.cat(logits_list)
hidden_states = torch.cat(hidden_states_list)

filename = 'llama-405'
llama_prompt_name = 'llama_asd'
with open(f'cache/hidden_states_{filename}/{prompt_name}_{llama_prompt_name}_logits_list.pt', 'wb') as f:
    torch.save(logits, f)

with open(f'cache/hidden_states_{filename}/{prompt_name}_{llama_prompt_name}_hidden_states_list.pt', 'wb') as f:
    torch.save(hidden_states, f)

"""
logits_list = []
for i in range(len(prompts)):
    if i % 1 == 0:
        print(i)
    with model.trace(prompts[i], remote=True) as runner:
        logits = model.lm_head.output.save()
        hidden_states = [model.model.layers[-1].output[0].save() for ilayer in [-50,-20,-10,-5,-2,-1]]
    print(hidden_states)
    exit()
    logits = logits[:, -1, :]
    logits = torch.nn.functional.softmax(logits, dim=-1)
    logits_list.append(logits)

num_processed = len(logits_list)
print('num_processed', num_processed)
for i in range(num_processed, len(prompts)):
    if i % 1 == 0:
        print(i)
    with model.trace(prompts[i], remote=True) as runner:
        logits = model.lm_head.output.save()
    logits = logits[:, -1, :]
    logits = torch.nn.functional.softmax(logits, dim=-1)
    logits_list.append(logits)

with open('cache/hidden_states_{}/{}_{}_logits_list.pkl'.format(filename, prompt_name, llama_prompt_name), 'wb') as f:
    pickle.dump(logits_list, f)
"""

