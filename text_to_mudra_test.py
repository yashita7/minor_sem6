import pandas as pd
import ast
import re
from deep_translator import GoogleTranslator
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.data.path.append("./nltk_data")

lemmatizer = WordNetLemmatizer()

# ===============================
# LOAD DATASETS
# ===============================

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
# ===============================
# BUILD SEMANTIC DICTIONARIES
# ===============================

mudra_dict = {}
face_dict = {}


def expand_word(word, lemmatizer):
    """
    Expand a word using:
    - original
    - noun lemma
    - verb lemma
    - WordNet synonyms
    """

    expanded = set()

    word = word.lower()
    expanded.add(word)

    # noun lemma
    expanded.add(lemmatizer.lemmatize(word, pos='n'))

    # verb lemma
    expanded.add(lemmatizer.lemmatize(word, pos='v'))

    # synonyms
    for syn in wordnet.synsets(word)[:2]:
        for l in syn.lemmas():
            expanded.add(l.name().lower().replace("_", " "))

    return expanded


# ---------------------------
# MUDRA DICTIONARY
# ---------------------------

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


# ---------------------------
# FACIAL EXPRESSION DICTIONARY
# ---------------------------

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


# ===============================
# TEXT PREPROCESSING
# ===============================

def preprocess(text):

    text = text.lower()

    words = re.findall(r'\b\w+\b', text)

    processed = []

    for w in words:

        lemma_n = lemmatizer.lemmatize(w, pos='n')
        lemma_v = lemmatizer.lemmatize(w, pos='v')
        lemma_a = lemmatizer.lemmatize(w, pos='a')

        processed.extend([w, lemma_n, lemma_v, lemma_a])

    # return list(set(processed))
    return processed


# ===============================
# TRANSLATION
# ===============================

def translate_if_needed(text):

    try:
        translated = GoogleTranslator(
            source='auto', target='en').translate(text)
        return translated
    except:
        return text


# ===============================
# SENTENCE → GESTURES
# ===============================

def sentence_to_gestures(sentence):

    translated = translate_if_needed(sentence)
    words = preprocess(translated)

    mudras = []
    expressions = []

    seen_mudra = set()
    seen_face = set()

    for word in words:

        # normalize emotion words
        if word in emotion_normalization:
            word = emotion_normalization[word]

        if word in mudra_dict:

            m = mudra_dict[word]

            if m not in seen_mudra:
                mudras.append(m)
                seen_mudra.add(m)

        if word in face_dict:

            f = face_dict[word]

            if f not in seen_face:
                expressions.append(f)
                seen_face.add(f)
    return translated, mudras, expressions


# ===============================
# TEST SENTENCES
# ===============================

# test_sentences = [

#     # Mudra focused
#     "The river flows in the forest",
#     "The king holds a crown",
#     "A flower is offered to god",


#     # Emotion + mudra
#     "A flower is offered with love",
#     "The warrior shows anger",
#     "The child laughs with joy",
#     "having fun in the rain with havy rainy clouds falling in love",

#     # Facial expressions
#     "The girl feels happy",
#     "The boy is joyful",
#     "The woman is sad",
#     "The man becomes angry",
#     "The child is scared",
#     "The devotee feels peaceful",
#     "The hero stands brave",
#     "The person feels disgust",

#     # Hindi tests
#     "राजा के सिर पर मुकुट है",
#     "नदी जंगल में बहती है",
#     "बच्चा खुश है",
#     "वह डर गया",
#     "वह गुस्से में है"
# ]

test_sentences = [

    # -----------------
    # Simple mudra tests
    # -----------------
    "The river flows in the forest",
    "The king holds a crown",
    "A flower is offered to god",
    "The bird flies across the sky",
    "The snake moves through the grass",
    "The moon shines in the night sky",
    "The wind blows across the river",
    "The warrior holds his weapon",
    "The devotee folds hands in prayer",

    # -----------------
    # Mudra + emotion
    # -----------------
    "A flower is offered with love",
    "The warrior shows anger",
    "The child laughs with joy",
    "Having fun in the rain with heavy clouds and falling in love",
    "The girl feels shy and smiles softly",
    "The boy becomes afraid in the dark forest",
    "The devotee feels peaceful during prayer",
    "The hero stands brave before the enemy",
    "The person feels disgust after seeing something dirty",

    # -----------------
    # Facial emotion tests
    # -----------------
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

    # -----------------
    # Small story tests
    # -----------------
    "A young girl walks through the forest and sees a beautiful flower near the river. She smiles with joy and offers the flower to god with devotion.",
    
    "A brave warrior enters the battlefield with anger and determination. He raises his weapon and challenges the enemy with great strength.",
    
    "A child plays happily near the river while birds fly in the sky and the wind moves through the trees.",
    
    "The devotee walks slowly into the temple, folds his hands in prayer, and offers flowers with deep devotion and peace in his heart.",
    
    "A frightened traveler moves through a dark forest at night. The wind blows strongly and strange sounds make him feel terrified.",

    # -----------------
    # Mythological style
    # -----------------
    "Lord Shiva stands with great power holding the trident while the moon shines on his head and the river flows from his hair.",
    
    "Krishna plays the flute near the river while cows gather around him and the gopis watch with love and devotion.",
    
    "Garuda flies across the sky with great strength while the sun shines brightly and the wind moves the clouds.",

    # -----------------
    # Longer narrative
    # -----------------
    "In the quiet forest a young devotee walks slowly toward the temple carrying flowers. The river flows nearby and the moon shines softly in the night sky. With love and devotion the devotee offers the flowers to god and feels deep peace.",

    "A powerful king sits proudly on his throne wearing a crown while warriors stand beside him. The people gather in respect and offer gifts with honor and loyalty.",

    # -----------------
    # Hindi tests
    # -----------------
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


# ===============================
# INTERACTIVE MODE
# ===============================

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
