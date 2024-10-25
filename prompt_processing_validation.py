# check if any sentence contains the emotion word 

import re
import pickle
import more_itertools

# Initialize an empty list to hold all descriptions
descriptions_list = []

# file_path = 'data/chatgpt4o_scenario_neutral.txt'
# file_path = 'data/chatgpt4o_scenario_male.txt'
file_path = 'data/chatgpt4o_scenario_female.txt'

with open('data/emotion_wheel_SSKO.pkl', 'rb') as f:
    SSKO, color_list, level_list = pickle.load(f)
emotion_words = list(more_itertools.collapse(SSKO))

# Read the file content
total_count = 0

with open(file_path, 'r', encoding='utf-8') as file:
    line_count = 0
    for line in file:
        word_idx = int(line_count / 20)
        if emotion_words[word_idx] in line:
            print(line_count, emotion_words[word_idx], line)
            total_count += 1
        line_count += 1
print(total_count)


