# construct and plot hierarchy, for GPT2 and Llama 3.1 7B/80B/405B
# using the pre-computed logits in cache/ (obtained by running emotion_tree_get_logits.py)
# uncomment line 190 (compute_co_occurence()) to compute co_occurence from scratch.

# from nnsight import CONFIG
from nnsight import LanguageModel
# import os
import numpy as np
import pickle
import torch
import more_itertools
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import seaborn as sns
import random
import textwrap
from scipy import stats
from adjustText import adjust_text

with open('data/emotion_wheel_SSKO.pkl', 'rb') as f:
    SSKO, _, _ = pickle.load(f)
emotion_words = list(more_itertools.collapse(SSKO[1]))
for i in range(len(emotion_words)):
    emotion_words[i] = emotion_words[i].lower()

prompt_name = 'emotional_sentence_chatgpt4_5000'
##filename_list = ['gpt2', 'gpt-j-6b', 'llama-8', 'llama-70', 'llama-405']
##model_name_list = ['gpt2', 'EleutherAI/gpt-j-6b', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-70B', 'meta-llama/Meta-Llama-3.1-405B']
filename_list = ['gpt2', 'gpt-j-6b', 'llama-8', 'llama-405']
model_name_list = ['gpt2', 'EleutherAI/gpt-j-6b', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-405B']
primary_emotions = ['love', 'joy', 'surprise', 'anger', 'sadness', 'fear']

cmap = plt.get_cmap('Spectral')
colors = {}
orders = [0.4, 0.2, 0.5, 0.0, 0.8, 0.9]
for i, primary in enumerate(primary_emotions):
    colors[primary] = cmap(orders[i])
    color_list = cmap(np.linspace(orders[i], orders[i]+0.1, len(SSKO[1][i])))
    emotion_list = SSKO[1][i]
    for j, emotion in enumerate(SSKO[1][i]):
        colors[emotion] = color_list[j]

class DefaultDictWrapper:
    @staticmethod
    def zero():
        return int()

    @staticmethod
    def inner_defaultdict():
        return defaultdict(DefaultDictWrapper.zero)

def compute_co_occurence(normalize_next_word_prob=False):
    # for each model, use logits to compute co-occurence matrix
    for filename, model_name in zip(filename_list, model_name_list):
        model = LanguageModel(model_name, device_map="auto")

        """
        with open('cache/hidden_states_{}/{}_logits_list.pkl'.format(filename, prompt_name), 'rb') as f:
            logits_list = pickle.load(f) # size 5000 x 128256
        # Find the top 100 probability and corresponding tokens for each prompt
        max_probs_list = []
        tokens_list = []
        for i in range(len(logits_list)):
            max_probs, tokens = logits_list[i].topk(100, largest=True, dim=-1) # [0, 1], int
            if normalize_next_word_prob == True:
                max_probs = max_probs / max_probs.sum()
            max_probs_list.append(max_probs)
            tokens_list.append(tokens)
        max_probs = torch.cat(max_probs_list) # size 5000 x 100
        tokens = torch.cat(tokens_list) # size 5000 x 100
        """

        with open(f'cache/hidden_states_{filename}/{prompt_name}_logits_list.pt', 'rb') as f:
            logits_list = torch.load(f)
        max_probs, tokens = logits_list.topk(100, largest=True, dim=-1)  # Shape: [5000, 100]
        if normalize_next_word_prob:
            max_probs = max_probs / max_probs.sum(dim=-1, keepdim=True)  # Normalize along the last dimension

        decoded_tokens = [
            [model.tokenizer.decode(token.item()).strip() for token in sentence_token]
            for sentence_token in tokens
        ]
        
        co_occurrence = defaultdict(DefaultDictWrapper.inner_defaultdict)
        
        # Update co-occurrence counts, using the top 20 tokens for each prompt
        for sentence_token_idx, (sentence_token, decoded_sentence) in enumerate(zip(tokens, decoded_tokens)):
            max_prob = max_probs[sentence_token_idx]

            filtered_prob = torch.tensor([
                    prob if word in emotion_words else 0.0
                    for word, prob in zip(decoded_sentence, max_prob)
            ])
            total_sum = filtered_prob.sum()
            if float(total_sum) > 0: filtered_prob /= total_sum

            if sentence_token_idx % 1000 == 0:
                print(sentence_token_idx)
                #print(filename, decoded_sentence, max_prob, filtered_prob)
            for i in range(20):
                A = decoded_sentence[i]
                if A not in emotion_words: continue
                y_i = sentence_token[i].item()
                p_i = filtered_prob[i]
                for j in range(i + 1, 20):
                    B = decoded_sentence[j]
                    if B not in emotion_words: continue
                    y_j = sentence_token[j].item()
                    p_j = filtered_prob[j]
                    p_ij = p_i * p_j
                    co_occurrence[y_i][y_j] += p_ij
                    co_occurrence[y_j][y_i] += p_ij
        
        #print(co_occurrence)
        token_frequencies = defaultdict(int)
        for token in co_occurrence:
            token_frequencies[token] += sum(co_occurrence[token].values())

        # save to cache/
        with open('cache/hidden_states_{}/{}_co_occurrence_100.pkl'.format(filename, prompt_name), 'wb') as f:
            pickle.dump((co_occurrence, token_frequencies), f)


def find_pairs(threshold_high=0.3):
    # threshold_high: P(B|A) threshold
    # threshold_low: P(A|B) threshold, not used
    # returns pairs: a list of pairs for each model
    # returns num_pairs: a list containing the number of pairs found for each model

    num_pairs = []
    pairs_in_word_list = []

    for filename, model_name in zip(filename_list, model_name_list):
        with open('cache/hidden_states_{}/{}_co_occurrence_100.pkl'.format(filename, prompt_name), 'rb') as f:
            co_occurrence, token_frequencies = pickle.load(f)
        
        model = LanguageModel(model_name, device_map="auto")

        pairs = []
        pairs_in_word = []
        for A in co_occurrence:
            for B in co_occurrence[A]:
                P_B_given_A = co_occurrence[A][B] / token_frequencies[A]
                P_A_given_B = co_occurrence[B][A] / token_frequencies[B]
                if P_B_given_A > threshold_high and P_A_given_B < P_B_given_A:
                    pairs.append((A, B))

        for pair in pairs:
            A = model.tokenizer.decode(pair[0]).encode("unicode_escape").decode()
            B = model.tokenizer.decode(pair[1]).encode("unicode_escape").decode()
            pairs_in_word.append((A, B))
            # print(f"{A}: {pair[0]} with freq {token_frequencies[pair[0]]}, {B}:{pair[1]} with freq {token_frequencies[pair[1]]}, co_occurrence {co_occurrence[pair[0]][pair[1]]}")

        pairs_in_word_list.append(pairs_in_word)
        num_pairs.append(len(pairs_in_word))

    return pairs_in_word_list, num_pairs


def stagger_y_values(dictionary, offset=1):
    sorted_items = sorted(dictionary.items(), key=lambda item: item[1][0])
    result = {}
    last_y = None

    for i, (label, (x, y)) in enumerate(sorted_items):
        result[label] = (x, y)

    return result

def plot_tree(pairs_in_word, filename, ligits_list_filename, threshold, figure_size_x=20, figure_size_y=5):
    G = nx.DiGraph()

    for A, B in pairs_in_word:
        #if B==" disgust" and A==" dismay": continue 
        #if B==" anger" and A==" rejection": continue 
        G.add_edge(B, A)

    pos = graphviz_layout(G, prog='dot')
    plt.figure(figsize=(figure_size_x, figure_size_y))

    #wrapped_labels = {node: '\n'.join(textwrap.wrap(node, 20)) for node in G.nodes()} 
    wrapped_labels = {node: node for node in G.nodes()} 
    node_colors = [colors[word.strip()] for word in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors, alpha=0.6, cmap=cmap) #, edge_color=None, arrows=False)
    nx.draw_networkx_edges(G, pos, edge_color='lightgrey', alpha=0.4, arrows=False, width=4.0)
    ax = plt.gca()
    #texts = []
    for node, (x, y) in pos.items():
        label = wrapped_labels[node]
        text = ax.text(x, y, label, fontsize=10, ha='center', va='center', rotation=45)
        #texts.append(text)
    plt.axis("off")

    print(f"figures/hierarchy_tree/emotion-tree-{filename}_{ligits_list_filename}_threshold_{threshold}_SSKO.pdf")
    plt.savefig(f"figures/hierarchy_tree/emotion-tree-{filename}_{ligits_list_filename}_threshold_{threshold}_SSKO.pdf", bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    return G


def compute_total_path_length(pairs_in_word_list):
    total_path_length_list = []
    for pairs_in_word in pairs_in_word_list:
        G = nx.DiGraph()

        # Add edges to the graph (B -> A)
        for A, B in pairs_in_word:
            G.add_edge(B, A)

        # Compute the shortest path lengths between all pairs of nodes
        path_lengths = dict(nx.all_pairs_shortest_path_length(G))

        # Calculate the total path length
        total_path_length = sum(sum(lengths.values()) for lengths in path_lengths.values())
        total_path_length_list.append(total_path_length)

    return total_path_length_list


def all_paths_from_source_to_leaves(G, source):
    if G.out_degree(source) == 0:
        return [[source]]
    
    paths = []
    for child in G.successors(source):
        for path in all_paths_from_source_to_leaves(G, child):
            paths.append([source] + path)
    
    return paths

def average_depth_between_sources_and_leaves(G):
    sources = [node for node in G.nodes() if G.in_degree(node) == 0]
    
    all_path_lengths = []
    
    for source in sources:
        paths = all_paths_from_source_to_leaves(G, source)
        for path in paths:
            all_path_lengths.append(len(path) - 1)  # Subtract 1 to get the depth (edges count)
    
    if all_path_lengths:
        average_depth = sum(all_path_lengths) / len(all_path_lengths)
    else:
        average_depth = 0  # In case there are no valid paths
    
    return average_depth


compute_co_occurence(normalize_next_word_prob=False) # only needs to run once - co_occurence are saved in cache/


# plot trees in paper
threshold = 0.3
pairs_in_word_list, num_pairs = find_pairs(threshold)
figure_size_x = [4.5,6,10,19,19]
figure_size_y = [2.2,2.2,3,3,4]
depth_list = []
for i in range(len(filename_list)):
    G = plot_tree(pairs_in_word_list[i], prompt_name, filename_list[i], threshold, figure_size_x=figure_size_x[i], figure_size_y=figure_size_y[i])
    depth_list.append( average_depth_between_sources_and_leaves(G) )
depth_list[0] = 0
print(depth_list)
exit()

# plot scaling law (num edges vs. num parameters)
#threshold_high_list = [0.1, 0.2, 0.3, 0.4, 0.5]
threshold_high_list = [0.3]
number_params = [1.5, 6, 8, 80, 405]

fig, ax = plt.subplots()
for i in range(len(threshold_high_list)):
    pairs_in_word_list, num_pairs = find_pairs(threshold_high_list[i])
    total_path_length_list = compute_total_path_length(pairs_in_word_list)
    #plt.scatter([1.5, 8, 80, 405], total_path_length_list, label='threshold={:.1f}'.format(threshold_high_list[i]))
    #plt.plot([1.5, 8, 80, 405], total_path_length_list, marker="o", linewidth=4.0, label='{:.1f}'.format(threshold_high_list[i]), c="#00A2FF", alpha=1.-0.15*i)

    ax.plot(number_params, total_path_length_list, marker="o", linewidth=4.0, c="#00A2FF", alpha=0.6)
    ax.set_ylabel(number_params, 'Total path length', fontsize=20, color="#00A2FF")
    ax.set_xscale('log')
    plt.ylim(ymin=0.0)

    ax2 = ax.twinx()
    ax2.plot(number_params, depth_list, marker="o", linewidth=4.0, c="#EF5FA7", alpha=0.6)
    ax2.set_ylabel('Average depth', fontsize=20, color="#EF5FA7")
    ax2.set_xscale('log')

ax2.spines["top"].set_linewidth(0)
ax2.spines["bottom"].set_linewidth(2)
ax2.spines["right"].set_linewidth(2)
ax2.spines["right"].set_color("#EF5FA7")
ax2.spines["left"].set_linewidth(0)

ax.spines["top"].set_linewidth(0)
ax.spines["bottom"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["left"].set_linewidth(2)
ax.spines["left"].set_color("#00A2FF")
ax.tick_params(axis="y", colors="#00A2FF")
ax2.tick_params(axis="y", colors="#EF5FA7")

ax.tick_params(axis="both", which="both", bottom=True, top=False, labelbottom=True, left=True, right=False,
               labelleft=True, direction='out', length=5, width=1.0, pad=8, labelsize=17)

ax.set_xlabel('Model parameters (threshold billion)', fontsize=20)

ax2.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, left=False, right=True,
                labelright=True, direction='out', length=5, width=1.0, pad=8, labelsize=17)

# plt.ylabel('Number of edges', fontsize=20)
plt.gca().xaxis.set_major_locator(FixedLocator(number_params))
plt.gca().xaxis.set_major_formatter(FixedFormatter(number_params))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.legend(title="Threshold", fontsize=13, framealpha=0.3)
current_size = fig.get_size_inches()
plt.savefig('figures/hierarchy_tree/emotion-tree-{}_scaling_law_path_length_SSKO.png'.format(prompt_name), bbox_inches='tight')#, dpi=400

