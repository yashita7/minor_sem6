import warnings
import pandas as pd
import ast
import re
from deep_translator import GoogleTranslator
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy

# -------------------- SETUP --------------------
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

nltk.data.path.append("./nltk_data")

# -------------------- LOAD DATA --------------------
print("Loading datasets...\n")

mudra_df = pd.read_csv("./data/mudras.csv")
face_df = pd.read_csv("./data/facial_expressions.csv")

mudra_df["meanings"] = mudra_df["meanings"].apply(ast.literal_eval)
mudra_df["viniyoga"] = mudra_df["viniyoga"].apply(ast.literal_eval)

face_df["meanings"] = face_df["meanings"].apply(ast.literal_eval)
face_df["viniyoga"] = face_df["viniyoga"].apply(ast.literal_eval)

print("Mudra dataset:", len(mudra_df))
print("Facial dataset:", len(face_df))

# -------------------- NORMALIZATION --------------------
emotion_normalization = {
    "happy": "joy", "joyful": "joy",
    "sad": "sadness",
    "angry": "anger",
    "afraid": "fear", "scared": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "peaceful": "peace",
    "love": "love"
}

# Navarasa mapping
emotion_to_rasa = {
    "joy": "Hasya",
    "sadness": "Karuna",
    "anger": "Raudra",
    "fear": "Bhayanaka",
    "disgust": "Bibhatsa",
    "surprise": "Adbhuta",
    "love": "Shringara",
    "peace": "Shanta"
}

MYTHOLOGICAL = ["krishna", "shiva", "rama", "durga", "ganesha", "vishnu"]

# -------------------- WORD EXPANSION --------------------


def expand_word(word):
    expanded = set()

    word = word.lower().strip()
    if not word:
        return expanded

    expanded.add(word)
    expanded.add(lemmatizer.lemmatize(word, 'n'))
    expanded.add(lemmatizer.lemmatize(word, 'v'))

    try:
        for syn in wordnet.synsets(word)[:2]:
            for l in syn.lemmas():
                expanded.add(l.name().lower().replace("_", " "))
    except:
        pass

    return expanded


# -------------------- BUILD DICTIONARIES --------------------
mudra_dict = {}
face_dict = {}

for _, row in mudra_df.iterrows():
    gesture = row["transliteration"]

    for meaning in row["meanings"]:
        for w in expand_word(meaning):
            mudra_dict[w] = gesture

    for phrase in row["viniyoga"]:
        for w in re.findall(r'\b\w+\b', phrase.lower()):
            for ew in expand_word(w):
                mudra_dict[ew] = gesture

for _, row in face_df.iterrows():
    gesture = row["transliteration"]

    for meaning in row["meanings"]:
        for w in expand_word(meaning):
            face_dict[w] = gesture

    for phrase in row["viniyoga"]:
        for w in re.findall(r'\b\w+\b', phrase.lower()):
            for ew in expand_word(w):
                face_dict[ew] = gesture

print("\nMappings built:")
print("Mudra:", len(mudra_dict))
print("Face:", len(face_dict))

# -------------------- PREPROCESS --------------------


def preprocess(text):
    words = re.findall(r'\b\w+\b', text.lower())
    processed = []

    for w in words:
        processed.extend([
            w,
            lemmatizer.lemmatize(w, 'n'),
            lemmatizer.lemmatize(w, 'v'),
            lemmatizer.lemmatize(w, 'a')
        ])

    return processed

# -------------------- TRANSLATE --------------------


def translate(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

# -------------------- ENTITY + CONTEXT --------------------


def get_entity_contexts(sentence):
    doc = nlp(sentence)
    entity_context = {}

    for token in doc:
        if token.dep_ in ["nsubj", "dobj", "pobj"]:
            entity = token.text

            context_tokens = set()
            context_tokens.add(token)
            context_tokens.add(token.head)

            for child in token.head.children:
                context_tokens.add(child)

            for child in token.children:
                context_tokens.add(child)

            context = " ".join([t.text for t in sorted(
                context_tokens, key=lambda x: x.i)])
            entity_context[entity] = context

    # fallback
    if not entity_context:
        entity_context["scene"] = sentence

    return entity_context

# -------------------- MAIN FUNCTION --------------------


def analyze_sentence(sentence):

    translated = translate(sentence)
    entity_contexts = get_entity_contexts(translated)

    results = {}

    for entity, context in entity_contexts.items():

        words = preprocess(context)

        mudra_scores = {}
        face_scores = {}

        # ENTITY TYPE
        priority = "primary" if entity.lower() in MYTHOLOGICAL else "secondary"

        for word in words:

            # normalize
            if word in emotion_normalization:
                word = emotion_normalization[word]

                if word in emotion_to_rasa:
                    rasa = emotion_to_rasa[word]
                    face_scores[rasa] = face_scores.get(rasa, 0) + 2

            # mudra scoring
            if word in mudra_dict:
                m = mudra_dict[word]
                mudra_scores[m] = mudra_scores.get(m, 0) + 1

            # facial scoring
            if word in face_dict:
                f = face_dict[word]
                face_scores[f] = face_scores.get(f, 0) + 1

        # select top results
        mudras = sorted(mudra_scores, key=mudra_scores.get, reverse=True)[:3]
        expressions = sorted(
            face_scores, key=face_scores.get, reverse=True)[:3]

        results[entity] = {
            "priority": priority,
            "context": context,
            "mudras": mudras,
            "expressions": expressions
        }

    return translated, results


# -------------------- TEST --------------------
print("\n==============================")
print("Testing")
print("==============================\n")

test_sentences = [

    # ======================
    # BASIC OBJECT + ACTION
    # ======================
    "The river flows in the forest",
    "The king wears a crown",
    "The bird flies in the sky",
    "The snake crawls on the ground",
    "The sun shines brightly",
    "The moon glows in the night",
    "The wind blows strongly",
    "The child plays near the river",
    "The girl walks through the garden",
    "The man sits on the throne",

    # ======================
    # EMOTIONS ONLY
    # ======================
    "The boy is happy",
    "The girl feels sad",
    "The man is very angry",
    "The child is scared",
    "The woman feels peaceful",
    "The student is surprised",
    "The person feels disgust",
    "The mother shows love",
    "The hero is brave",
    "The devotee feels calm",

    # ======================
    # ACTION + EMOTION
    # ======================
    "The warrior fights with anger",
    "The child laughs with joy",
    "The girl smiles with love",
    "The boy cries in sadness",
    "The devotee prays peacefully",
    "The man shouts in anger",
    "The frightened child runs away",
    "The hero stands bravely",
    "The woman looks surprised",
    "The person reacts with disgust",

    # ======================
    # DEVOTIONAL / TEMPLE
    # ======================
    "The devotee offers flowers to god",
    "The devotee folds hands in prayer",
    "A man prays in the temple",
    "A woman offers diya to god",
    "The भक्त prays with devotion",
    "The devotee sings bhajans",
    "The priest performs पूजा",
    "The girl offers flowers with love",
    "The man bows before god",
    "The devotee feels peace in the temple",

    # ======================
    # NATURE SCENES
    # ======================
    "The river flows beside the mountain",
    "Birds fly across the blue sky",
    "The wind moves the trees",
    "The rain falls from dark clouds",
    "The sun rises in the east",
    "The moon shines over the river",
    "The forest is calm and silent",
    "The ocean waves move strongly",
    "Clouds cover the sky",
    "The stars shine at night",

    # ======================
    # MYTHOLOGICAL
    # ======================
    "Krishna plays the flute",
    "Lord Shiva holds the trident",
    "Rama shoots an arrow",
    "Krishna dances with gopis",
    "Shiva meditates in the mountains",
    "Rama fights the demon",
    "Krishna smiles with love",
    "Shiva shows great power",
    "Rama protects his people",
    "Krishna plays near the river",

    # ======================
    # MIXED COMPLEX SENTENCES
    # ======================
    "A happy child plays near the river while birds fly in the sky",
    "A brave warrior enters the battlefield with anger and strength",
    "The devotee walks into the temple and offers flowers with devotion",
    "A frightened boy walks through the dark forest at night",
    "The girl sees a flower and smiles with joy",
    "The man becomes angry and shouts loudly",
    "The child laughs and runs happily",
    "The woman cries with sadness",
    "The hero fights bravely and wins",
    "The devotee feels peace and love",

    # ======================
    # LONG STORY TESTS
    # ======================
    "A young girl walks through the forest and sees a beautiful flower near the river. She smiles with joy and offers the flower to god.",
    
    "A brave warrior enters the battlefield with anger. He raises his weapon and challenges the enemy with strength.",
    
    "A child plays happily near the river while birds fly in the sky and the wind moves the trees.",
    
    "The devotee walks slowly into the temple, folds hands in prayer, and offers flowers with devotion.",
    
    "A frightened traveler walks through a dark forest at night and feels scared.",
    
    "Krishna plays the flute near the river while cows gather and people watch with love.",
    
    "Lord Shiva stands with power holding the trident while the moon shines on his head.",
    
    "A king sits on his throne wearing a crown while people offer respect and gifts.",
    
    "A peaceful village lies near the river where children play and birds sing.",
    
    "A storm begins with strong winds, dark clouds, and heavy rain",

    # ======================
    # HINDI TESTS
    # ======================
    "राजा सिंहासन पर बैठा है",
    "नदी बह रही है",
    "बच्चा खुश है",
    "लड़का डर गया",
    "वह गुस्से में है",
    "भक्त भगवान को फूल चढ़ाता है",
    "लड़की मंदिर में पूजा करती है",
    "हवा तेज चल रही है",
    "चाँद रात में चमकता है",
    "सूरज उग रहा है",

    # ======================
    # EDGE CASES (IMPORTANT)
    # ======================
    "Love and anger both exist in the heart",
    "The person feels nothing",
    "Silence fills the empty forest",
    "The boy stands still",
    "The girl looks around",
    "An unknown figure appears in the dark",
    "Everything is calm and quiet",
    "Chaos and fear spread everywhere",
    "A sudden surprise shocks everyone",
    "The man feels confused and lost"
]

for s in test_sentences:
    translated, results = analyze_sentence(s)

    print("Input:", s)
    print("Translated:", translated)

    for entity, data in results.items():
        print(f"\nEntity: {entity} ({data['priority']})")
        print("Context:", data["context"])
        print("Mudras:", " → ".join(
            data["mudras"]) if data["mudras"] else "None")
        print("Expressions:", " → ".join(
            data["expressions"]) if data["expressions"] else "None")

    print("--------------------------------------------------")

# -------------------- INTERACTIVE --------------------
print("\n===== Dance Generator =====")

while True:
    user_input = input("\nEnter sentence (or 'exit'): ")

    if user_input.lower() == "exit":
        break

    translated, results = analyze_sentence(user_input)

    print("\nEnglish:", translated)

    for entity, data in results.items():
        print(f"\nEntity: {entity} ({data['priority']})")
        print("Context:", data["context"])
        print("Mudras:", " → ".join(
            data["mudras"]) if data["mudras"] else "None")
        print("Expressions:", " → ".join(
            data["expressions"]) if data["expressions"] else "None")

print("\nProgram finished.")
