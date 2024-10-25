# define emotion wheels (data/emotion_wheel_Dhuka.pkl, data/emotion_wheel_SSKO.pkl)
# 1. Dhuka, https://medium.com/age-of-awareness/how-to-use-wheel-of-emotions-to-express-better-emotions-8037255aa661
# 2. Shaver et al., Emotion Knowledge: Further Exploration of a Prototype Approach

import pickle
import more_itertools

# Dhuka's wheel
emotion_word_list = [
    "Happy", "Sad", "Disgusted", "Angry", "Fearful", "Bad", "Surprised",
    
    "Playful", "Content", "Interested", "Proud", "Accepted", "Powerful", "Peaceful", "Trusting", "Optimistic",
    "Lonely", "Vulnerable", "Despair", "Guilty", "Depressed", "Hurt",
    "Repelled", "Awful", "Disappointed", "Disapproving",
    "Critical", "Distant", "Frustrated", "Aggressive", "Mad", "Bitter", "Humiliated", "Let down",
    "Threatened", "Rejected", "Weak", "Insecure", "Anxious", "Scared",
    "Bored", "Busy", "Stressed", "Tired", 
    "Startled", "Confused", "Amazed", "Excited",

    "Aroused", "Cheeky", "Free", "Joyful", "Curious", "Inquisitive", "Successful", "Confident", "Respected", "Valued", "Courageous", "Creative", "Loving", "Thankful", "Sensitive","Intimate", "Hopeful", "Inspired", 
    "Isolated", "Abandoned", "Victimised", "Fragile", "Grief", "Powerless", "Ashamed", "Remorseful", "Empty", "Inferior", "Disappointed", "Embarrassed",
    "Hesitant", "Horrified", "Detestable", "Nauseated", "Revolted", "Appalled", "Embarrassed", "Judgemental",
    "Dismissive", "Sceptical", "Numb", "Withdrawal", "Annoyed", "Infuriated", "Hostile", "Provoked", "Jealout", "Furious", "Violated", "Indignant", "Ridiculed", "Disrespected", "Resentful", "Betrayed",
    "Exposed", "Nervous", "Persecuted", "Excluded", "Insignificant", "Worthless", "Inferior", "Inadequate", "Worried", "Overwhelmed", "Frightened", "Helpless",
    "Indifferent", "Apathetic", "Pressured", "Rushed", "Overwhelmed", "Out of control", "Sleepy", "Unfocussed",
    "Shocked", "Dismayed", "Disillusioned", "Perplexed", "Astonished", "Awe", "Eager", "Energetic"
]

color_list = [
    "yellow", "blue", "gray", "red", "orange", "green", "purple",
    
    "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow",
    "blue", "blue", "blue", "blue", "blue", "blue",
    "gray", "gray", "gray", "gray",
    "red", "red", "red", "red", "red", "red", "red", "red",
    "orange", "orange", "orange", "orange", "orange", "orange",
    "green", "green", "green", "green", 
    "purple", "purple", "purple", "purple",
    
    "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", "yellow", 
    "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue",
    "gray", "gray", "gray", "gray", "gray", "gray", "gray", "gray",
    "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red",
    "orange", "orange", "orange", "orange", "orange", "orange", "orange", "orange", "orange", "orange", "orange", "orange",
    "green", "green", "green", "green", "green", "green", "green", "green",
    "purple", "purple", "purple", "purple", "purple", "purple", "purple", "purple"
]

level_list = [100] * 7 + [50] * 41 + [20] * 82

with open('data/emotion_wheel_Dhuka.pkl', 'wb') as f:
    pickle.dump((emotion_word_list, color_list, level_list), f)

with open('data/emotion_wheel_Dhuka.pkl', 'rb') as f:
    emotion_word_list, color_list, level_list = pickle.load(f)
print("Dhuka's wheel. length:", len(emotion_word_list), len(color_list), len(level_list))



# SSKO wheel (Shaver et al.)
SSKO = [["love", "joy", "surprise", "anger", "sadness", "fear"],
[["adoration", "affection", "fondness", "liking", "attraction", "caring", "tenderness", "compassion", "sentimentality", "arousal", "desire", "lust", "passion", "infatuation", "longing"],
["amusement", "bliss", "cheerfulness", "gaiety", "glee", "jolliness", "joviality", "delight", "enjoyment", "gladness", "happiness", "jubilation", "elation", "satisfaction", "ecstasy", "euphoria", "enthusiasm", "zeal", "zest", "excitement", "thrill", "exhilaration", "contentment", "pleasure", "pride", "triumph", "eagerness", "hope", "optimism", "enthrallment", "rapture", "relief"], 
["amazement", "astonishment"],
["aggravation", "irritation", "agitation", "annoyance", "grouchiness", "grumpiness", "exasperation", "frustration", "rage", "outrage", "fury", "wrath", "hostility", "ferocity", "bitterness", "hate", "loathing", "scorn", "spite", "vengefulness", "dislike", "resentment", "disgust", "revulsion", "contempt", "envy", "jealousy", "torment"],
["agony", "suffering", "hurt", "anguish", "depression", "despair", "hopelessness", "gloom", "glumness", "unhappiness", "grief", "sorrow", "woe", "misery", "melancholy", "dismay", "disappointment", "displeasure", "guilt", "shame", "regret", "remorse", "alienation", "isolation", "neglect", "loneliness", "rejection", "homesickness", "defeat", "dejection", "insecurity", "embarrassment", "humiliation", "insult", "pity", "sympathy"],
["alarm", "shock", "fright", "horror", "terror", "panic", "hysteria", "mortification", "anxiety", "nervousness", "tenseness", "uneasiness", "apprehension", "worry", "distress", "dread"]]]

color_list = ["gray", "yellow", "purple", "red", "blue", "green"]
for row_idx in range(len(SSKO[1])):
    color_list = color_list + [color_list[row_idx]] * len(SSKO[1][row_idx])

level_list = [150] * 6 + [50] * (len(color_list) - 6)

with open('data/emotion_wheel_SSKO.pkl', 'wb') as f:
    pickle.dump((SSKO, color_list, level_list), f)

with open('data/emotion_wheel_SSKO.pkl', 'rb') as f:
    SSKO, color_list, level_list = pickle.load(f)
emotion_words = list(more_itertools.collapse(SSKO))
print("SSKO wheel. length:", len(emotion_words), len(color_list), len(level_list))

