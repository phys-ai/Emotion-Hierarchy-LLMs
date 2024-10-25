import re

# Initialize an empty list to hold all descriptions
descriptions_list = []

# file_path = 'data/chatgpt4o_scenario_neutral_unprocessed.txt'
# save_filename = 'data/chatgpt4o_scenario_neutral.txt'

# file_path = 'data/chatgpt4o_scenario_male_unprocessed.txt'
# save_filename = 'data/chatgpt4o_scenario_male.txt'

file_path = 'data/chatgpt4o_scenario_female_unprocessed.txt'
save_filename = 'data/chatgpt4o_scenario_female.txt'

# Read the file content
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Remove any standalone '---' to prevent interference
content = content.replace('---', '')

# sections = re.split(r'###\s+\d+\s+Descriptions\s+of\s+([\w\s]+)\s+Scenarios', content) # neutral
# sections = re.split(r'###\s+\*\*\d+\s+Stories\s+of\s+(.+?)\*\*', content) # male
sections = re.split(r'###\s+\d+\s+Stories\s+of\s+(.+)', content) # female

# The split will result in a list where even indices are content before the first header
# and odd indices are the captured emotion names. We need to process them in pairs.
# Example: [pre_content, emotion1, section1_content, emotion2, section2_content, ...]

# Iterate through the sections list in steps of 2 (emotion name and its content)
for i in range(1, len(sections), 2):
    emotion = sections[i].strip()
    section_content = sections[i + 1].strip()
    
    # Debug: Print the current emotion being processed
    print(f'Processing Emotion: {emotion}')
    
    # Attempt to find numbered descriptions
    numbered_descriptions = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', section_content, re.DOTALL)
    
    if numbered_descriptions:
        # Process numbered descriptions
        for desc in numbered_descriptions:
            # Remove any leading/trailing whitespace and replace newlines with spaces
            clean_desc = desc.strip().replace('\n', ' ')
            descriptions_list.append(clean_desc)
    else:
        # Handle unnumbered descriptions
        # Split the section into paragraphs based on two or more newlines
        paragraphs = re.split(r'\n{2,}', section_content)
        for desc in paragraphs:
            clean_desc = desc.strip().replace('\n', ' ')
            if clean_desc:
                descriptions_list.append(clean_desc)

# Optional: Verify the total number of descriptions
print(f'Total Descriptions Extracted: {len(descriptions_list)}')

# Optionally, save the list to a Python file for later use
with open(save_filename, 'w', encoding='utf-8') as outfile:
    for desc in descriptions_list:
        outfile.write(desc)
        outfile.write('\n')
