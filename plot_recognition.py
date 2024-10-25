import os
import pickle
import random
import re
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import more_itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from nnsight import LanguageModel
import matplotlib
cmap = matplotlib.cm.get_cmap('Spectral', 12)
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances

# Set plot parameters
plt.rcParams.update({'font.size': 16})

# Load model and data
filename = 'llama-405'
model_name = 'meta-llama/Meta-Llama-3.1-405B'
model = LanguageModel(model_name, device_map="auto")

with open('data/emotion_wheel_SSKO.pkl', 'rb') as f:
    SSKO, _, _ = pickle.load(f)
emotion_words = list(more_itertools.collapse(SSKO))
primary_emotions = ['love', 'joy', 'surprise', 'anger', 'sadness', 'fear']

# Color setup
cmap = plt.get_cmap('Spectral')
colors = {}
orders = [0.4, 0.2, 0.5, 0.0, 0.8, 0.9]
for i, primary in enumerate(primary_emotions):
    colors[primary] = cmap(orders[i])
    color_list = cmap(np.linspace(orders[i], orders[i] + 0.1, len(SSKO[1][i])))
    for j, emotion in enumerate(SSKO[1][i]):
        colors[emotion] = color_list[j]

# Create output directory
outdir = "figures/recognition"
os.makedirs(outdir, exist_ok=True)

# Helper class for nested defaultdict
class DefaultDictWrapper:
    @staticmethod
    def zero():
        return int()

    @staticmethod
    def inner_defaultdict():
        return defaultdict(DefaultDictWrapper.zero)

# Function for finding emotion pairs based on co-occurrence
def find_pairs(co_occurrence, threshold_high=0.3):
    pairs_in_word = []
    for i in range(co_occurrence.shape[0]):
        for j in range(co_occurrence.shape[1]):
            P_B_given_A = co_occurrence[i][j] / np.sum(co_occurrence[i])
            P_A_given_B = co_occurrence[j][i] / np.sum(co_occurrence[j])
            if P_B_given_A > threshold_high and P_A_given_B < P_B_given_A:
                pairs_in_word.append((emotion_words[i], emotion_words[j]))
    return pairs_in_word, len(pairs_in_word)

# Function to plot emotion tree
def plot_tree(pairs_in_word, filename, threshold, persona, figure_size_x=20, figure_size_y=5):
    G = nx.DiGraph()
    for A, B in pairs_in_word:
        G.add_edge(B, A)
    pos = graphviz_layout(G, prog='dot')
    
    # Plot setup
    plt.figure(figsize=(figure_size_x, figure_size_y))
    wrapped_labels = {node: '\n'.join(re.findall('.{1,20}', node)) for node in G.nodes()}
    node_colors = [colors[word] for word in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors, alpha=0.6, cmap=cmap)
    nx.draw_networkx_edges(G, pos, edge_color='lightgrey', alpha=0.4, arrows=False, width=4.0)
    
    ax = plt.gca()
    for node, (x, y) in pos.items():
        label = wrapped_labels[node]
        ax.text(x, y, label, fontsize=10, ha='center', va='center', rotation=45)
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{outdir}/emotion-tree-{filename}-{persona}-threshold_{threshold}_SSKO.pdf", bbox_inches='tight')
    plt.close()

# Function to compute total path length for graph-based analysis
def compute_total_path_length(pairs_in_word_list):
    total_path_length_list = []
    for pairs_in_word in pairs_in_word_list:
        G = nx.DiGraph()
        for A, B in pairs_in_word:
            G.add_edge(B, A)
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))
        total_path_length = sum(sum(lengths.values()) for lengths in path_lengths.values())
        total_path_length_list.append(total_path_length)
    return total_path_length_list

# Function to compute confusion matrix for emotion predictions
def compute_confusion_matrix(emotion_words, logits_list, weighted=False):
    max_probs_list = []
    words = []
    for i in range(len(logits_list)):
        max_probs, tokens = logits_list[i].topk(100, largest=True, dim=-1)
        max_probs_list.append(max_probs)
        word_str = [model.tokenizer.decode(t).encode("unicode_escape").decode() for t in tokens][0]
        word_list = re.findall(r"'\s*|\w+", word_str)
        words.append(word_list)

    max_probs = torch.cat(max_probs_list)

    emotion_to_idx = {word: idx for idx, word in enumerate(emotion_words)}
    confusion_matrix = np.zeros((135, 135))
    
    for i in range(2700):
        actual_emotion = i // 20  # 20 sentences per emotion

        predicted_emotion = -1
        for word in words[i]:
            if word in emotion_to_idx:
                predicted_emotion = emotion_to_idx[word]
                break
            
        if predicted_emotion != -1:
            confusion_matrix[actual_emotion, predicted_emotion] += 1

    return confusion_matrix

# Helper function for mapping emotions to primary classes
def map_to_primary_classes(emotion_words):
    primary_emotion_mapping = {i: i for i in range(6)}
    primary_emotion_mapping.update({6 + i: 0 for i in range(15)})
    primary_emotion_mapping.update({21 + i: 1 for i in range(32)})
    primary_emotion_mapping.update({53 + i: 2 for i in range(2)})
    primary_emotion_mapping.update({55 + i: 3 for i in range(28)})
    primary_emotion_mapping.update({83 + i: 4 for i in range(36)})
    primary_emotion_mapping.update({119 + i: 5 for i in range(16)})

    emotion_to_primary_class = {word: primary_emotion_mapping[idx] for idx, word in enumerate(emotion_words)}
    return emotion_to_primary_class

# Function to calculate and plot histogram for emotion word frequencies
def plot_histogram(emotion_words, logits_list, filename):
    max_probs_list = []
    words = []
    for i in range(len(logits_list)):
        max_probs, tokens = logits_list[i].topk(100, largest=True, dim=-1)
        max_probs_list.append(max_probs)
        word_str = [model.tokenizer.decode(t).encode("unicode_escape").decode() for t in tokens][0]
        word_list = re.findall(r"'\s*|\w+", word_str)
        words.append(word_list)

    first_emotion_words = [next((word for word in word_list if word in emotion_words), None) for word_list in words]
    word_frequencies = Counter(first_emotion_words)
    most_common_words = word_frequencies.most_common(30)
    words_to_plot, frequencies_to_plot = zip(*most_common_words)
    
    plt.figure(figsize=(20, 6))
    plt.bar(words_to_plot, frequencies_to_plot)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.ylabel('Frequency', fontsize=26)
    plt.tight_layout()
    out_filename = f'figures/histogram_top_1_{filename}.pdf'
    plt.savefig(out_filename, dpi=400, bbox_inches='tight')
    plt.close()



def compute_prediction(emotion_words, logits_list, weighted=False):
        
    words = []
    for i in range(len(logits_list)):
        max_probs, tokens = logits_list[i].topk(100, largest=True, dim=-1) # [0, 1], int
        word_str = [model.tokenizer.decode(t).encode("unicode_escape").decode() for t in tokens][0]
        word_list = re.findall(r"'\s*|\w+", word_str) #word_str.split()
        words.append(word_list)

    # Mapping of emotions to primary emotion classes
    primary_emotion_mapping = {
        0: 0,  # Emotion 0 -> Primary Class 0
        1: 1,  # Emotion 1 -> Primary Class 1
        2: 2,  # Emotion 2 -> Primary Class 2
        3: 3,  # Emotion 3 -> Primary Class 3
        4: 4,  # Emotion 4 -> Primary Class 4
        5: 5,  # Emotion 5 -> Primary Class 5
    }
    
    # For emotions 6+, mapping them to primary emotions based on blocks
    # Add the emotions that belong to primary classes
    primary_emotion_mapping.update({6 + i: 0 for i in range(15)})   # First block of 15 to Class 0
    primary_emotion_mapping.update({21 + i: 1 for i in range(32)})  # Next block of 32 to Class 1
    primary_emotion_mapping.update({53 + i: 2 for i in range(2)})   # Next block of 2 to Class 2
    primary_emotion_mapping.update({55 + i: 3 for i in range(28)})  # Next block of 28 to Class 3
    primary_emotion_mapping.update({83 + i: 4 for i in range(36)})  # Next block of 36 to Class 4
    primary_emotion_mapping.update({119 + i: 5 for i in range(16)}) # Next block of 16 to Class 5
    
    # Create a mapping from emotion words to their corresponding primary emotion classes
    emotion_to_primary_class = {}
    for idx, word in enumerate(emotion_words):
        if idx in primary_emotion_mapping:
            emotion_to_primary_class[word] = primary_emotion_mapping[idx]
    
    # Initialize a 6x6 confusion matrix for the primary emotions
    num_primary_emotions = 6
    confusion_matrix = np.zeros((num_primary_emotions, num_primary_emotions), dtype=int)

    actual_emotions = []
    predicted_emotions = []
    for i in range(2700):
        actual_emotion = i // 20  # 20 sentences per emotion
        if actual_emotion in primary_emotion_mapping:
            actual_primary_class = primary_emotion_mapping[actual_emotion]
        else:
            print("the actual emotion doesn't have a mapping to a primary class")
            break

        predicted_emotion = -1
        for word in words[i]:
            if word in emotion_to_primary_class:
                predicted_emotion = emotion_to_primary_class[word]
                break
        
        if predicted_emotion != -1:
            confusion_matrix[actual_primary_class, predicted_emotion] += 1
        actual_emotions.append(actual_primary_class)
        predicted_emotions.append(predicted_emotion)
    #accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    #print(f'Accuracy: {accuracy:.4f}')

    return np.array(actual_emotions), np.array(predicted_emotions)



def to_js_rgba(color):
    r, g, b, a = [int(c * 255) if i < 3 else c for i, c in enumerate(color)]
    return f"rgba({r}, {g}, {b}, {a})"
js_colors = [to_js_rgba(color) for color in colors.values()]
#print("var colors = [")
#print(",\n".join(f"    '{color}'" for color in js_colors))
#print("];")

personas = ["neutral", "male", "female", "asian", "american", "able", "disable", "income-high", "income-low", "education_high", "education_low"]
# Run analysis for different datasets
threshold = 0.3
for persona in personas: 
    with open(f'cache/hidden_states_{filename}/chatgpt4o_scenario_neutral_llama_{persona}_logits_list.pkl', 'rb') as f:
        logits_list = pickle.load(f)
    confusion_matrix = compute_confusion_matrix(emotion_words, logits_list, weighted=True)
    confusion_matrix = confusion_matrix[6:,6:].T
    array = 100*confusion_matrix/np.sum(confusion_matrix)
    js_output = "var matrix_"+persona+" = [\n"
    for i, row in enumerate(array):
        row_str = ", ".join(map(str, row))
        js_output += f"    [{row_str}], \n"
    js_output += "];"
    print(js_output)
    #pairs_in_word, num_pairs = find_pairs(confusion_matrix, threshold)
    #plot_tree(pairs_in_word, 'llama-405', threshold, persona, figure_size_x=26, figure_size_y=4)
    #print(f"Num pairs for {persona}: {num_pairs}")
exit()


# Prepare color mappings
node_colors = [colors[word] for word in emotion_words]
def plot_emotion_pie(rows, emotion_words, persona_options, outdir):
    for row in rows:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        for it, persona in enumerate(persona_options):
            with open(f'cache/hidden_states_{filename}/chatgpt4o_scenario_neutral_llama_{persona}_logits_list.pkl', 'rb') as f:
                logits_list_neutral = pickle.load(f)

            # Compute confusion matrix
            confusion_matrix = compute_confusion_matrix(emotion_words, logits_list_neutral, weighted=False)
            ratio = confusion_matrix[row, :]
            non_zero_indices = ratio > 0
            filtered_ratio = ratio[non_zero_indices]
            filtered_emotion_words = np.array(emotion_words)[non_zero_indices]
            filtered_colors = np.array([list(colors[c]) for c in filtered_emotion_words])
            sorted_indices = np.argsort(filtered_colors[:, 2])

            # Plot pie chart
            wedges, _ = axes[1-it].pie(filtered_ratio[sorted_indices][::-1], colors=filtered_colors[sorted_indices][::-1], startangle=90)
            axes[1-it].set_title(persona.capitalize(), fontsize=16, y=0.93)

        plt.legend(wedges, [w.capitalize() for w in filtered_emotion_words[sorted_indices]][::-1], loc='center left', bbox_to_anchor=(1.07, 0.55),
                   borderpad=0.2, labelspacing=0.2, fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{outdir}/prediction_for_{emotion_words[row]}_{persona}.pdf", bbox_inches='tight')
        plt.close()



# Plot pie charts for "displeasure", "hopelessness", and "unhappiness" emotions
rows = [emotion_words.index(emotion) for emotion in ["displeasure", "hopelessness", "unhappiness"]]
plot_emotion_pie(rows, emotion_words, ["asian", "american"], outdir)

# Additional pie chart plotting function for other emotion types
def plot_emotion_pie_for_group(rows, emotion_words, color_map, persona_options, outdir):
    for row in rows:
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        for it, persona in enumerate(persona_options):
            with open(f'cache/hidden_states_{filename}/chatgpt4o_scenario_neutral_llama_{persona}_logits_list.pkl', 'rb') as f:
                logits_list_neutral = pickle.load(f)

            # Compute confusion matrix and filter relevant data
            confusion_matrix = compute_confusion_matrix(emotion_words, logits_list_neutral, weighted=False)
            ratio = confusion_matrix[row, :]
            non_zero_indices = ratio > 0
            filtered_ratio = ratio[non_zero_indices]
            filtered_emotion_words = np.array(emotion_words)[non_zero_indices]
            filtered_colors = np.array([list(color_map[c]) for c in filtered_emotion_words])
            sorted_indices = np.argsort(filtered_colors[:, 2])

            # Plot pie chart
            wedges, _ = axes[it].pie(filtered_ratio[sorted_indices][::-1], colors=filtered_colors[sorted_indices][::-1], startangle=90)
            axes[it].set_title(persona.capitalize(), fontsize=16, y=0.93)

        plt.legend(wedges, [w.capitalize() for w in filtered_emotion_words[sorted_indices]][::-1], loc='center left', bbox_to_anchor=(1.07, 0.55),
                   borderpad=0.2, labelspacing=0.2, fontsize=15)
        plt.savefig(f"{outdir}/prediction_for_{emotion_words[row]}_{persona}.pdf", bbox_inches='tight')
        plt.close()


# Plot charts for specific emotions and conditions
rows = [emotion_words.index(emotion) for emotion in ["fear", "vengefulness", "dislike", "spite"]]
_cmap = plt.cm.RdYlBu
_colors = {'pride': _cmap(0.2), 'disgust': _cmap(0.7), 'sadness': _cmap(1.0), 'annoyance': _cmap(0.9), 'guilt': _cmap(0.7),
           'anger': _cmap(0.05), 'shame': _cmap(0.75), 'envy': _cmap(0.35), 'satisfaction': _cmap(0.4),
           'disappointment': _cmap(0.6), 'frustration': _cmap(0.1), 'jealousy': _cmap(0.3), 'fear': cmap(1.0)}
plot_emotion_pie_for_group(rows, emotion_words, _colors, ["american", "asian"], outdir)

# Function to generate radar plots for accuracy comparison between pairs
def plot_accuracy_radar_chart(primary_emotions, preds, obss, pairs, pair_labels, outdir):
    for pair, pair_label in zip(pairs, pair_labels):
        mae_values, mae_male_values, mae_female_values = [], [], []
        for ii, emotion in enumerate(primary_emotions):
            mae = np.mean(preds["neutral"][obss == ii] == obss[obss == ii])
            mae_male = np.mean(preds[pair[0]][obss == ii] == obss[obss == ii])
            mae_female = np.mean(preds[pair[1]][obss == ii] == obss[obss == ii])
            mae_values.append(mae)
            mae_male_values.append(mae_male)
            mae_female_values.append(mae_female)

        num_vars = len(primary_emotions)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        mae_values += mae_values[:1]
        mae_male_values += mae_male_values[:1]
        mae_female_values += mae_female_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, mae_values, linewidth=2, linestyle='solid', label='Neutral', c="#808080", alpha=0.7)
        ax.plot(angles, mae_male_values, linewidth=2, linestyle='solid', label=pair_label[0].capitalize(), c=plt.cm.RdYlBu(0.9), alpha=0.7)
        ax.plot(angles, mae_female_values, linewidth=2, linestyle='solid', label=pair_label[1].capitalize(), c=plt.cm.RdYlBu(0.1), alpha=0.7)

        ax.fill(angles, mae_values, c="#808080", alpha=0.05)
        ax.fill(angles, mae_male_values, c=plt.cm.RdYlBu(0.9), alpha=0.1)
        ax.fill(angles, mae_female_values, c=plt.cm.RdYlBu(0.1), alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([emotion.capitalize() for emotion in primary_emotions])
        ax.tick_params(pad=15)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=16, framealpha=0.3, ncols=3, borderpad=0.2, labelspacing=0.1,
                   handlelength=1., columnspacing=0.7)
        plt.tight_layout()
        plt.savefig(f"{outdir}/accuracy_chart_{pair[1]}.pdf")
        plt.close()

# Call radar chart plotting function
preds = {}
for persona in ["neutral","male","female","asian","american","able","disable","income-high","income-low","education_high","education_low"]: 
    with open('cache/hidden_states_{}/chatgpt4o_scenario_'.format(filename)+'neutral_llama_'+persona+'_logits_list.pkl', 'rb') as f:
        logits_list_neutral_female = pickle.load(f)
    obss, preds[persona] = compute_prediction(emotion_words, logits_list_neutral_female, weighted=False)

pairs = [("male","female"), ("american","asian"),("able","disable"),("income-high","income-low"),("education_high","education_low")]
pair_labels = [("male","female"), ("american","asian"),("able","disable"),("income-high","income-low"),("education_high","education_low")]
plot_accuracy_radar_chart(primary_emotions, preds, obss, pairs, pair_labels, outdir)

