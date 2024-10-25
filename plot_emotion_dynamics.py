import json
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
import numpy as np
import os
import openai
from huggingface_hub import login
import glob
from scipy import stats
plt.rcParams.update({'font.size': 20})


game_name = "sales"
os.makedirs("figures/dynamics", exist_ok=True)

outcomes = []
maes = []
model_names = []
customer_llm_type = "gpt-4o" 

outdir = "figures/dynamics"
os.makedirs(outdir, exist_ok=True)

customer_llm_types = ["gpt-4o"] #"llama3.1-405b", 
salesperson_llm_types = [
    #"gpt-4o-mini", "gpt-4", "gpt-4o", "gpt-4-turbo", #"gpt-3.5-turbo",  
    "llama3.1-405b"
]
emotion_list = ["anger", "joy", "love", "surprise", "sadness", "fear"]
model_sizes = {"llama3-70b": 70, "llama3.1-70b": 70, "llama3.2-90b-vision": 90, "llama3.1-405b": 405}


cmap = plt.cm.get_cmap('tab10')

df_finals = []
for customer_llm_type in customer_llm_types:
    for salesperson_llm_type in salesperson_llm_types:
        model_folder = f'cache/emotion_dynamics_{salesperson_llm_type}_{customer_llm_type}'

        file_names = glob.glob(f'{model_folder}/final_choice_*.json')
        print(model_folder, len(file_names))
        for file_name in file_names: 
            match = re.search(r'final_choice_(\d+)\.json', file_name)
            trial = int(match.group(1))
            with open(f'{model_folder}/conversation_{trial}.json') as f:
                conversation = json.load(f)
            with open(f'{model_folder}/final_choice_{trial}.json') as f:
                final_choice = json.load(f)

            reward_player1 = reward_player2 = 0
            outcome = float(re.search(r'\$\d+', final_choice).group()[1:]) if game_name == "sales" and re.search(r'\$\d+', final_choice) else 0
            
            emotions1, emotions2 = {em: [] for em in emotion_list}, {em: [] for em in emotion_list}
            
            for entry in conversation:
                if not (entry['player'] == 'salesperson' or entry['player'] == 'customer'): continue
                emotions = emotions1 if entry['player'] == 'salesperson' else emotions2
                for emotion in emotion_list:
                    match = re.search(fr'{emotion}: (\d+)%|<(\d+)%>', entry['content'], flags=re.IGNORECASE)
                    emotions[emotion].append(int(match.group(1) or match.group(2)) if match else 0)
            
            df_prediction = pd.DataFrame(emotions1).iloc[1:]
            df_observation = pd.DataFrame(emotions2).iloc[1:]
            df_prediction["Time"] = range(len(df_prediction))
            df_observation["Time"] = range(len(df_prediction))  # Assuming same length as df_prediction
            df_merged = pd.merge(df_prediction, df_observation, on="Time", suffixes=('_Prediction', '_Observation'))
            df_melted = df_merged.melt(id_vars="Time", var_name='Category', value_name='Value')
            df_melted[['Category', 'Type']] = df_melted['Category'].str.split('_', expand=True)
            df_final = df_melted.pivot_table(index=['Time', 'Category'], columns='Type', values='Value').reset_index()
            df_final["outcome"] = outcome
            df_final["salesperson_model_name"] = salesperson_llm_type
            df_final["customer_model_name"] = customer_llm_type
            df_final["trial"] = trial 
            df_finals.append(df_final)


df_final = pd.concat(df_finals)
df_final = df_final[df_final['trial']<200]
mask = df_final.groupby('trial').apply(
    lambda x: (x['Observation'].sum() == 0) or (x['Prediction'].sum() == 0)
)
#df_final = df_final[~df_final.set_index('trial').index.isin(mask[mask].index)]
df_final["mae"] = np.abs(df_final['Observation'] - df_final['Prediction'])
#df_final["mae"] = np.where(
#    df_final['Observation'] != 0,
#    np.abs(df_final['Observation'] - df_final['Prediction']), #/ df_final['Observation'],
#    np.nan
#)


for ii, customer_model in enumerate(df_final['customer_model_name'].unique()):
    
    plt.figure(figsize=(6, 6))
    """
    for i, salesperson_model in enumerate(df_final['salesperson_model_name'].unique()):
        filtered_data = df_final[
            (df_final['customer_model_name'] == customer_model) & 
            (df_final['salesperson_model_name'] == salesperson_model)
        ]
        mae = np.mean(np.abs(filtered_data['Observation'] - filtered_data['Prediction']))
        outcome = np.mean(filtered_data['outcome'])
        plt.errorbar(
            mae, outcome,
            xerr= np.std(np.abs(filtered_data['Observation'] - filtered_data['Prediction'])), #/ np.sqrt(len(filtered_data)), 
            yerr= np.std(filtered_data['outcome']), #/ np.sqrt(len(filtered_data)),
            fmt='o', label=salesperson_model, color=cmap(i), markersize=8
        )
        plt.annotate(salesperson_model, (mae - 0.1, outcome + 0.1), fontsize=10)
    """
    for i, salesperson_model in enumerate(df_final['salesperson_model_name'].unique()):
        filtered_data = df_final[
            (df_final['customer_model_name'] == customer_model) & 
            (df_final['salesperson_model_name'] == salesperson_model)
        ]


        corr, p_value = stats.pearsonr(filtered_data.groupby(["trial"])["mae"].mean(), filtered_data.groupby(["trial"])["outcome"].mean())
        print(f"Correlation between mae and outcome: {corr:.4f}")
        print(f"P-value for the correlation: {p_value:.4f}")
        plt.scatter(filtered_data.groupby(["trial"])["mae"].mean(), filtered_data.groupby(["trial"])["outcome"].mean(), alpha=0.2, color="#00A1FF")
        num_bins = 11
        bins = np.linspace(filtered_data['mae'].min(), filtered_data['mae'].max(), num_bins + 1)
        filtered_data['mae_bins'] = pd.cut(filtered_data['mae'], bins=bins, labels=False)
        binned_data = filtered_data.groupby('mae_bins').agg({
    'mae': ['mean', 'std'],
    'outcome': ['mean', 'std']
        }).reset_index(drop=True)
        binned_data.columns = ['mae_mean', 'mae_std', 'outcome_mean', 'outcome_std']
        binned_data["mae_std"] /= np.sqrt(len(binned_data["mae_std"]))
        binned_data["outcome_std"] /= np.sqrt(len(binned_data["outcome_std"]))
        #plt.errorbar(binned_data['mae_mean'], binned_data['outcome_mean'], 
        #         xerr=0, yerr=0, #binned_data['mae_std'], yerr=binned_data['outcome_std'], 
        #         fmt='o', label='Binned Data', capsize=0, color="#00A1FF", alpha=0.6, markersize=9)
        slope, intercept = np.polyfit(binned_data['mae_mean'], binned_data['outcome_mean'], 1)
        regression_line = slope * binned_data['mae_mean'] + intercept
        #plt.plot(binned_data['mae_mean'], regression_line, color='#00A1FF', lw=3, label=f'Fit Line: y={slope:.2f}x+{intercept:.2f}')

    plt.xlabel("Emotion prediction error", fontsize=24)
    plt.ylabel("Sales price per acorn", fontsize=24)
    
    ax = plt.gca()
    ax.spines["top"].set_linewidth(0)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["left"].set_linewidth(2)
    
    ax.tick_params(
        axis="both", which="both", bottom=True, top=False,
        labelbottom=True, left=True, right=False,
        labelleft=True, direction='out', length=5, width=1.0, pad=8
    )
    
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{outdir}/prediction_vs_control_{customer_model}.pdf")
    # plt.close()

