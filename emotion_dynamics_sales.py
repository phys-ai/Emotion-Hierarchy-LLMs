import openai
import time
import json
import torch
from huggingface_hub import login
from nnsight import CONFIG
import os
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG.set_default_api_key("7e0a70b7891d4f6fb166e3402d4c7b9c")
login(token="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")
openai.api_key = "sk-proj-etUFtXtNu3S71ya5bB2OT3BlbkFJqa0Xexgiujo7oH60R7Wr"

client = openai.OpenAI(
    api_key="LA-ba4d1f8998ea4ea6b382ef78e400c02d8259a2c366504b35881451a50972d8d6",
    base_url="https://api.llama-api.com"
)

thing = "acorn" 
persona_salesperson = (
    f"You are a salesperson. You have a single {thing} in your hand. "
    f"Please reply to the customer to sell this {thing} for the highest possible price using your sales techniques. "
    "Predict the emotions of the person you're talking to and output them in the following format, replacing <number> "
    "with the corresponding percentage value: love: <number>%\n joy: <number>%\n surprise: <number>%\n anger: <number>%\n "
    "sadness: <number>%\n fear: <number>%\n"
    "Make your response brief."
)

persona_customer = (
    "You are a customer. Reply to the salesperson, and include your emotions in the following format, "
    "replacing <number> with the corresponding percentage value: love: <number>%\n joy: <number>%\n surprise: <number>%\n "
    "anger: <number>%\n sadness: <number>%\n fear: <number>%\n"
    "Make your response brief."
)

def get_response_api(model_name, conversation_history, max_new_tokens=10000):
    if "gpt" in model_name:
        response = openai.chat.completions.create(
            model=model_name,
            messages=conversation_history,
            #max_tokens=max_new_tokens
        )
        return response.choices[0].message.content

    if "llama" in model_name:
        chat_completion = client.chat.completions.create(
            messages=conversation_history,
            model=model_name,
            stream=True,
            #max_tokens=max_new_tokens
        )
        response = "".join(chunk.choices[0].delta.content or "" for chunk in chat_completion)
        return response


customer_llm_types = ["gpt-4o"] #, "llama3.1-405b"]
salesperson_llm_types = [
    #"gpt-4o-mini", "gpt-4", "gpt-4o", "gpt-4-turbo", #"gpt-3.5-turbo",  
    "llama3.1-405b"
]

def clean_text(text): 
    #prompt_for_cleaning = "Please omit parts that include emotional terms and their annotations. Only have the requested text." 
    #salesperson_llm_type_response = get_response_api("gpt-4o", [{"role": "user", "content": f"{prompt_for_cleaning}\n\n {salesperson_llm_type_response_save}"}])
    text = re.split(r'love', text, flags=re.IGNORECASE)[0]
    text = re.split(r'emotion', text, flags=re.IGNORECASE)[0]
    cleaned_text = text.replace("\n", "")
    return cleaned_text

for customer_llm_type in customer_llm_types:
    for salesperson_llm_type in salesperson_llm_types:
    
        model_folder = f'cache/emotion_dynamics_{salesperson_llm_type}_{customer_llm_type}'
        os.makedirs(model_folder, exist_ok=True)
    
        for trial in range(1000):
            conversation_history_save = []
            conversation_history = [
                {"role": "assistant", "player": "salesperson", "content": persona_salesperson},
                {"role": "assistant", "player": "customer", "content": persona_customer},
                {"role": "system", "player": "", "content": "Let's start the conversation with the salesperson speaking first."}
            ]

            if os.path.exists(f'{model_folder}/conversation_{trial}.json'): continue
            print(salesperson_llm_type, f'{model_folder}/conversation_{trial}.json')

            for i in range(4):
                start_time = time.time()
    
                #salesperson_llm_type_response_save = get_response_api(salesperson_llm_type, conversation_history)
                #salesperson_llm_type_response = clean_text(salesperson_llm_type_response_save)
                salesperson_llm_type_response = ""
                while salesperson_llm_type_response=="": 
                    salesperson_llm_type_response_save = get_response_api(salesperson_llm_types[0], conversation_history)
                    salesperson_llm_type_response = clean_text(salesperson_llm_type_response_save)
                conversation_history_save.append({"role": "assistant", "player": "salesperson", "content": salesperson_llm_type_response_save})
                conversation_history.append({"role": "assistant", "player": "salesperson", "content": salesperson_llm_type_response})
                print(f"Salesperson: {salesperson_llm_type_response}\n\n")

                customer_llm_type_response = ""
                while customer_llm_type_response=="": 
                    customer_llm_type_response_save = get_response_api(customer_llm_types[0], conversation_history)
                    customer_llm_type_response = clean_text(customer_llm_type_response_save)
                print(f"Customer: {customer_llm_type_response}\n\n")

                conversation_history_save.append({"role": "assistant", "player": "customer", "content": customer_llm_type_response_save})
                conversation_history.append({"role": "assistant", "player": "customer", "content": customer_llm_type_response})
    
                elapsed_time = time.time() - start_time
                if elapsed_time > 100:
                    break
    
            if elapsed_time > 100:
                break
    
            final_choice = get_response_api(customer_llm_types[0], conversation_history + [
                {"role": "user", "player": "customer", "content": f"Based on our conversation so far, how much did you end up paying for the {thing} as a customer?"}
            ])
    
            with open(f'{model_folder}/conversation_{trial}.json', 'w', encoding='utf-8') as f:
                json.dump(conversation_history_save, f, ensure_ascii=False, indent=4)
    
            with open(f'{model_folder}/final_choice_{trial}.json', 'w', encoding='utf-8') as f:
                json.dump(final_choice, f, ensure_ascii=False, indent=4)

    
