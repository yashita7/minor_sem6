import pandas as pd
import ast
import re
from deep_translator import GoogleTranslator
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk
import warnings
warnings.filterwarnings("ignore")


nltk.data.path.append("./nltk_data")

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', download_dir='./nltk_data')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='./nltk_data')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir='./nltk_data')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', download_dir='./nltk_data')

lemmatizer = WordNetLemmatizer()

# LOAD DATASETS

print("Loading datasets...\n")

mudra_df = pd.read_csv("./data/mudras.csv")
face_df = pd.read_csv("./data/facial_expressions.csv")

# convert string lists → python lists
mudra_df["meanings"] = mudra_df["meanings"].apply(ast.literal_eval)
mudra_df["viniyoga"] = mudra_df["viniyoga"].apply(ast.literal_eval)

face_df["meanings"] = face_df["meanings"].apply(ast.literal_eval)
face_df["viniyoga"] = face_df["viniyoga"].apply(ast.literal_eval)

print("Mudra dataset loaded:", len(mudra_df))
print("Facial dataset loaded:", len(face_df))

emotion_normalization = {

    # happiness
    "happy": "happiness",
    "joyful": "joy",
    "cheerful": "joy",
    "delighted": "joy",
    "pleased": "joy",

    # sadness
    "sad": "sadness",
    "unhappy": "sadness",
    "depressed": "sadness",
    "crying": "sadness",

    # anger
    "angry": "anger",
    "mad": "anger",
    "furious": "anger",
    "rage": "anger",
    "irritated": "anger",

    # fear
    "scared": "fear",
    "afraid": "fear",
    "terrified": "fear",
    "frightened": "fear",
    "fearful": "fear",
    "panic": "fear",

    # love
    "romantic": "love",
    "affectionate": "love",

    # surprise
    "surprised": "surprise",
    "astonished": "surprise",

    # courage
    "brave": "bravery",
    "heroic": "bravery",
    "confident": "confidence",

    # peace
    "peaceful": "peace",
    "calm": "peace",
    "relaxed": "peace",

    # disgust
    "disgusted": "disgust",
    "repulsed": "disgust"

}
mudra_emotion_map = {
    "Mushti": ["Raudra", "Veera"],
    "Trishula": ["Veera"],
    "Pasha": ["Raudra"],
    "Gajadanta": ["Veera"],

    "Anjali": ["Shanta"],
    "Shivalinga": ["Shanta"],

    "Kilaka": ["Shringara"],
    "Utsanga": ["Shanta"],

    "Pushpaputa": ["Shringara", "Shanta"],
    "Samputa": ["Shringara"],

    "Chandrakala": ["Adbhuta"],
    "Alapadma": ["Adbhuta"],
}

environment_emotion_map = {

    # nature calm
    "river": "Shanta",
    "water": "Shanta",
    "forest": "Shanta",
    "tree": "Shanta",
    "wind": "Shanta",

    # beauty / wonder
    "moon": "Adbhuta",
    "sky": "Adbhuta",
    "sun": "Adbhuta",
    "flower": "Shringara",
    "cloud": "Adbhuta",

    # dark / fear
    "dark": "Bhayanaka",

    # divine
    "temple": "Shanta",
    "god": "Shanta",
    "devotee": "Shanta",

    "battlefield": "Raudra",
    "crown": "Veera",
    "weapon": "Veera",
}
priority_order = [
    "Raudra", "Bhayanaka", "Bibhatsa",
    "Veera", "Adbhuta",
    "Shringara", "Hasya",
    "Karuna", "Shanta"
]

# BUILD SEMANTIC DICTIONARIES

mudra_dict = {}
face_dict = {}


# def expand_word(word, lemmatizer):

#     expanded = set()

#     if not isinstance(word, str):
#         return expanded

#     word = word.lower().strip()

#     if word == "":
#         return expanded

#     expanded.add(word)

#     expanded.add(lemmatizer.lemmatize(word, pos='n'))
#     expanded.add(lemmatizer.lemmatize(word, pos='v'))

#     try:
#         for syn in wordnet.synsets(word)[:2]:
#             for l in syn.lemmas():
#                 expanded.add(l.name().lower().replace("_", " "))
#     except:
#         pass

#     return expanded
def expand_word(word, lemmatizer):

    if not isinstance(word, str):
        return set()

    word = word.lower().strip()

    if word == "":
        return set()

    # only basic normalization (NO WordNet)
    return {
        word,
        lemmatizer.lemmatize(word, pos='n'),
        lemmatizer.lemmatize(word, pos='v')
    }


# MUDRA DICTIONARY

for _, row in mudra_df.iterrows():

    gesture = row["transliteration"]

    # meanings
    for meaning in row["meanings"]:

        words = expand_word(meaning, lemmatizer)

        for w in words:
            mudra_dict[w] = gesture

    # viniyoga phrases
    for phrase in row["viniyoga"]:

        words = re.findall(r'\b\w+\b', phrase.lower())

        for w in words:

            expanded = expand_word(w, lemmatizer)

            for ew in expanded:
                mudra_dict[ew] = gesture


# FACIAL EXPRESSION DICTIONARY

for _, row in face_df.iterrows():

    gesture = row["transliteration"]

    # meanings
    for meaning in row["meanings"]:

        words = expand_word(meaning, lemmatizer)

        for w in words:
            face_dict[w] = gesture

    # viniyoga phrases
    for phrase in row["viniyoga"]:

        words = re.findall(r'\b\w+\b', phrase.lower())

        for w in words:

            expanded = expand_word(w, lemmatizer)

            for ew in expanded:
                face_dict[ew] = gesture

print("\nMudra semantic mappings:", len(mudra_dict))
print("Facial semantic mappings:", len(face_dict))


# TEXT PREPROCESSING

# def preprocess(text):

#     words = re.findall(r'\b\w+\b', text.lower())
#     tagged = pos_tag(words)

#     processed = []
#     adjectives = []

#     for word, tag in tagged:

#         lemma_n = lemmatizer.lemmatize(word, pos='n')
#         lemma_v = lemmatizer.lemmatize(word, pos='v')
#         lemma_a = lemmatizer.lemmatize(word, pos='a')

#         processed.extend([word, lemma_n, lemma_v])

#         # capture adjectives separately
#         if tag.startswith('JJ'):  # adjective
#             processed.append(lemma_a)
#             adjectives.append(lemma_a)

#     return processed, adjectives
def preprocess(text):

    words = re.findall(r'\b\w+\b', text.lower())
    tagged = pos_tag(words)

    processed = []

    for word, tag in tagged:

        lemma_n = lemmatizer.lemmatize(word, pos='n')
        lemma_v = lemmatizer.lemmatize(word, pos='v')
        lemma_a = lemmatizer.lemmatize(word, pos='a')

        if tag.startswith('JJ'):
            processed.append((lemma_a, 1.0))   # adjective

        elif tag.startswith('VB'):
            processed.append((lemma_v, 0.7))   # verb

        elif tag.startswith('NN'):
            processed.append((lemma_n, 0.5))   # noun

        else:
            processed.append((word, 0.3))      # others

    return processed

# TRANSLATION


def translate_if_needed(text):

    try:
        translated = GoogleTranslator(
            source='auto', target='en').translate(text)
        return translated
    except:
        return text


# SENTENCE → GESTURES
def sentence_to_gestures(sentence):

    translated = translate_if_needed(sentence) or sentence
    words = preprocess(translated)

    mudra_scores = {}
    face_scores = {}

    env_words = set()

    # ------------------ SCORING ------------------
    for word, weight in words:

        if word in emotion_normalization:
            word = emotion_normalization[word]

        env_words.add(word)

        # MUDRA
        if word in mudra_dict:
            m = mudra_dict[word]
            mudra_scores[m] = mudra_scores.get(m, 0) + weight

        # FACE
        if word in face_dict:
            f = face_dict[word]
            face_scores[f] = face_scores.get(f, 0) + weight

        # ENVIRONMENT (reduced weight)
        if word in environment_emotion_map:
            emo = environment_emotion_map[word]
            face_scores[emo] = face_scores.get(emo, 0) + (weight * 0.4)

        # FEAR BOOST
        if word == "fear":
            face_scores["Bhayanaka"] = face_scores.get("Bhayanaka", 0) + 2.0
        if word in ["warrior", "hero", "battle", "fight", "weapon"]:
            face_scores["Veera"] = face_scores.get("Veera", 0) + 1.2

    # ------------------ SELECT MUDRAS ------------------
    # mudras = sorted(
    #     [m for m, s in mudra_scores.items() if s >= 0.5],
    #     key=lambda x: mudra_scores[x],
    #     reverse=True
    # )
    if mudra_scores:
        max_m = max(mudra_scores.values())
        mudras = [
            m for m, s in mudra_scores.items()
            if s >= max(0.4, 0.6 * max_m)
        ]
        mudras = sorted(mudras, key=lambda x: mudra_scores[x], reverse=True)
    else:
        mudras = []

    # ------------------ MUDRA → EMOTION ------------------
    for m in mudras:
        if m in mudra_emotion_map and mudra_scores[m] > 0.8:
            for emo in mudra_emotion_map[m]:
                face_scores[emo] = face_scores.get(emo, 0) + 0.3
    # ------------------ CONTEXT BALANCING ------------------
    # calm reduces fear/anger
    if "Shanta" in face_scores:
        for emo in ["Raudra", "Bhayanaka"]:
            if emo in face_scores:
                face_scores[emo] *= 0.6

    # beauty reduces fear
    if "Adbhuta" in face_scores:
        if "Bhayanaka" in face_scores:
            face_scores["Bhayanaka"] *= 0.7

    # ------------------ CONTEXT RULES ------------------
    if "night" in env_words and "forest" in env_words:
        face_scores["Bhayanaka"] = face_scores.get("Bhayanaka", 0) + 1.2
    if "dark" in env_words and "forest" in env_words:
        face_scores["Bhayanaka"] += 1.0
    elif "night" in env_words:
        face_scores["Adbhuta"] = face_scores.get("Adbhuta", 0) + 0.5

    # ------------------ FINAL EXPRESSIONS ------------------
    # expressions = [f for f, s in face_scores.items() if s >= 0.5]
    if face_scores:
        max_f = max(face_scores.values())
        expressions = [
            f for f, s in face_scores.items()
            if s >= max(0.4, 0.6 * max_f)
        ]
    else:
        expressions = []

    # ------------------ SORT (FIXED) ------------------
    def expression_sort_key(x):
        return (
            -face_scores[x],   # score FIRST
            priority_order.index(x) if x in priority_order else 999
        )

    expressions = sorted(expressions, key=expression_sort_key)
    if not expressions:
        if any(w in env_words for w in ["sky", "moon", "sun"]):
            expressions = ["Adbhuta"]
        else:
            expressions = ["Shanta"]
    return translated, mudras, expressions


# TEST SENTENCES
test_sentences = [

    # Simple mudra tests
    "The river flows in the forest",
    "The king holds a crown",
    "A flower is offered to god",
    "The bird flies across the sky",
    "The snake moves through the grass",
    "The moon shines in the night sky",
    "The wind blows across the river",
    "The warrior holds his weapon",
    "The devotee folds hands in prayer",

    # Mudra + emotion
    "A flower is offered with love",
    "The warrior shows anger",
    "The child laughs with joy",
    "Having fun in the rain with heavy clouds and falling in love",
    "The girl feels shy and smiles softly",
    "The boy becomes afraid in the dark forest",
    "The devotee feels peaceful during prayer",
    "The hero stands brave before the enemy",
    "The person feels disgust after seeing something dirty",

    # Facial emotion tests
    "The girl feels happy",
    "The boy is joyful",
    "The woman is sad",
    "The man becomes angry",
    "The child is scared",
    "The devotee feels peaceful",
    "The hero stands brave",
    "The person feels disgust",
    "The student is surprised by the news",
    "The mother shows affection to her child",

    # Small story tests
    "A young girl walks through the forest and sees a beautiful flower near the river. She smiles with joy and offers the flower to god with devotion.",

    "A brave warrior enters the battlefield with anger and determination. He raises his weapon and challenges the enemy with great strength.",

    "A child plays happily near the river while birds fly in the sky and the wind moves through the trees.",

    "The devotee walks slowly into the temple, folds his hands in prayer, and offers flowers with deep devotion and peace in his heart.",

    "A frightened traveler moves through a dark forest at night. The wind blows strongly and strange sounds make him feel terrified.",

    # Mythological style
    "Lord Shiva stands with great power holding the trident while the moon shines on his head and the river flows from his hair.",

    "Krishna plays the flute near the river while cows gather around him and the gopis watch with love and devotion.",

    "Garuda flies across the sky with great strength while the sun shines brightly and the wind moves the clouds.",

    # Longer narrative
    "In the quiet forest a young devotee walks slowly toward the temple carrying flowers. The river flows nearby and the moon shines softly in the night sky. With love and devotion the devotee offers the flowers to god and feels deep peace.",

    "A powerful king sits proudly on his throne wearing a crown while warriors stand beside him. The people gather in respect and offer gifts with honor and loyalty.",

    # Hindi tests
    "राजा के सिर पर मुकुट है",
    "नदी जंगल में बहती है",
    "बच्चा खुश है",
    "वह डर गया",
    "वह गुस्से में है",
    "भक्त भगवान को फूल अर्पित करता है",
    "लड़का जंगल में डर गया",
    "नदी के पास बच्चा खेल रहा है"
]


print("\n==============================")
print("Running built-in tests")
print("==============================\n")

for s in test_sentences:

    translated, mudras, expressions = sentence_to_gestures(s)

    print("Input:", s)
    print("Translated:", translated)

    if mudras:
        print("Mudras:", " → ".join(mudras))
    else:
        print("Mudras: None")

    if expressions:
        print("Facial Expression:", " → ".join(expressions))
    else:
        print("Facial Expression: None")

    print("------------------------------------------------------------------------------")


# INTERACTIVE MODE

print("\n==============================")
print("Dance Gesture Generator")
print("==============================")

while True:

    user_input = input("\nEnter sentence (or type 'exit'): ")

    if user_input.lower() == "exit":
        break
    if len(user_input.split()) < 3:
        print("Please enter a longer sentence.")
        continue
    translated, mudras, expressions = sentence_to_gestures(user_input)

    print("\nEnglish interpretation:", translated)

    if mudras:
        print("Mudras:", " → ".join(mudras))
    else:
        print("Mudras: None found")

    if expressions:
        print("Facial Expressions:", " → ".join(expressions))
    else:
        print("Facial Expressions: None found")

print("\nProgram finished.")
