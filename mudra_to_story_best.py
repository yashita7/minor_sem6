import pandas as pd
import ast
import re
import random
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  LOAD DATASETS
# ─────────────────────────────────────────────────────────────

mudra_df = pd.read_csv("./data/mudras.csv")
face_df = pd.read_csv("./data/facial_expressions.csv")

mudra_df["meanings"] = mudra_df["meanings"].apply(ast.literal_eval)
mudra_df["viniyoga"] = mudra_df["viniyoga"].apply(ast.literal_eval)
face_df["meanings"] = face_df["meanings"].apply(ast.literal_eval)
face_df["viniyoga"] = face_df["viniyoga"].apply(ast.literal_eval)

# ─────────────────────────────────────────────────────────────
#  BUILD INFO DICTS FROM YOUR ACTUAL DATA
# ─────────────────────────────────────────────────────────────

mudra_info = {}
for _, row in mudra_df.iterrows():
    mudra_info[row["transliteration"]] = {
        "meanings":    row["meanings"],
        "viniyoga":    row["viniyoga"],
        "description": row["description"],
        "english":     row["english_name"],
        "category":    row["category"],
    }

rasa_info = {}
for _, row in face_df.iterrows():
    rasa_info[row["transliteration"]] = {
        "meanings":    row["meanings"],
        "viniyoga":    row["viniyoga"],
        "description": row["description"],
        "english":     row["english_name"],
    }

# ─────────────────────────────────────────────────────────────
#  RASA STORY VOCABULARY
# ─────────────────────────────────────────────────────────────

rasa_story = {
    "Hasya": {
        "settings": ["the sunlit riverside", "the open festival ground",
                     "the flower-filled courtyard", "the village square at dawn"],
        "tones":    ["playfully", "with laughter", "with delight", "joyfully"],
        "actions":  ["dances", "leaps", "spins", "claps hands", "plays"],
        "emotion":  ["joy", "delight", "laughter", "happiness"],
        "closing":  "The world feels light and bright.",
    },
    "Karuna": {
        "settings": ["the empty riverbank", "the still evening forest",
                     "the grey and silent path", "the darkened doorway"],
        "tones":    ["slowly", "with a heavy heart", "in silence", "sorrowfully"],
        "actions":  ["weeps", "kneels", "waits", "mourns", "longs"],
        "emotion":  ["grief", "sorrow", "loss", "compassion"],
        "closing":  "A deep sadness settles over everything.",
    },
    "Raudra": {
        "settings": ["the battlefield", "the burning ground",
                     "the storm-swept plain", "the crumbling fortress"],
        "tones":    ["fiercely", "with burning rage", "with great fury", "violently"],
        "actions":  ["strikes", "roars", "charges forward", "confronts", "challenges"],
        "emotion":  ["fury", "wrath", "rage", "anger"],
        "closing":  "The earth trembles beneath the weight of wrath.",
    },
    "Bhayanaka": {
        "settings": ["the dark forest at night", "the abandoned ruins",
                     "the foggy riverbank at midnight", "the shadowed cave"],
        "tones":    ["trembling", "with wide eyes", "breathlessly", "in terror"],
        "actions":  ["freezes", "hides", "steps back", "watches in dread", "trembles"],
        "emotion":  ["terror", "dread", "panic", "fear"],
        "closing":  "Every shadow holds a threat.",
    },
    "Shringara": {
        "settings": ["the moonlit river bank", "the blooming jasmine grove",
                     "the lotus pond at dawn", "the fragrant evening garden"],
        "tones":    ["gently", "softly", "tenderly", "with longing"],
        "actions":  ["gazes", "adorns", "reaches out", "offers", "moves gracefully"],
        "emotion":  ["love", "longing", "tenderness", "beauty"],
        "closing":  "The air is filled with sweetness.",
    },
    "Adbhuta": {
        "settings": ["beneath the vast night sky", "at the edge of the horizon",
                     "beside the glowing river", "on the mountain peak at dawn"],
        "tones":    ["in awe", "with wide eyes", "in wonder", "breathlessly"],
        "actions":  ["stares", "marvels", "stands still", "gazes upward", "witnesses"],
        "emotion":  ["wonder", "awe", "amazement", "reverence"],
        "closing":  "The world reveals something extraordinary.",
    },
    "Veera": {
        "settings": ["the open battlefield", "the royal hall",
                     "the mountain pass", "the gate of the great fortress"],
        "tones":    ["boldly", "with great strength", "fearlessly", "with determination"],
        "actions":  ["stands firm", "raises a hand", "commands", "protects", "defeats"],
        "emotion":  ["courage", "strength", "heroism", "honor"],
        "closing":  "Victory is written in every step.",
    },
    "Bibhatsa": {
        "settings": ["the forsaken ground", "the ruined shrine",
                     "the abandoned field", "the dark and rotting place"],
        "tones":    ["with revulsion", "turning away", "in disgust", "with horror"],
        "actions":  ["recoils", "turns away", "steps back", "covers face", "refuses"],
        "emotion":  ["disgust", "repulsion", "aversion", "rejection"],
        "closing":  "The sight is unbearable.",
    },
    "Shanta": {
        "settings": ["the quiet riverbank", "the temple courtyard",
                     "the still forest clearing at sunrise", "the garden of meditation"],
        "tones":    ["quietly", "in stillness", "peacefully", "with deep calm"],
        "actions":  ["sits", "closes eyes", "breathes slowly", "meditates", "prays"],
        "emotion":  ["peace", "calm", "serenity", "devotion"],
        "closing":  "Everything is still. Everything is enough.",
    },
}

# ─────────────────────────────────────────────────────────────
#  CHARACTER SELECTION — single entity
# ─────────────────────────────────────────────────────────────

rasa_characters = {
    "Raudra":    ["The warrior", "The fierce king", "The lone warrior"],
    "Veera":     ["The great warrior", "The king", "The brave soldier"],
    "Bhayanaka": ["The traveler", "The young devotee", "The wanderer"],
    "Shanta":    ["The devotee", "The sage", "The meditating sage"],
    "Shringara": ["The young woman", "The dancer", "She"],
    "Hasya":     ["The child", "The young dancer", "The joyful dancer"],
    "Karuna":    ["The woman", "The grieving devotee", "He"],
    "Adbhuta":   ["The sage", "The young devotee", "The lone wanderer"],
    "Bibhatsa":  ["The warrior", "The traveler", "He"],
}

# ─────────────────────────────────────────────────────────────
#  MUDRA → ROLE INFERENCE (NEW)
# ─────────────────────────────────────────────────────────────

mudra_role_map = {

    # royalty / power
    "Gajadanta": ["king", "royal figure"],
    "Chakra": ["king", "divine ruler"],
    "Shikhara": ["king", "leader"],

    # warrior / fight
    "Mushti": ["warrior", "fighter"],
    "Trishula": ["warrior", "shiva"],
    "Pasha": ["warrior", "enemy"],
    "Kartarimukha": ["warrior"],

    # devotion
    "Anjali": ["devotee"],
    "Pushpaputa": ["devotee"],
    "Kapota": ["devotee"],
    "Shivalinga": ["devotee"],

    # feminine / love
    "Alapadma": ["woman", "dancer"],
    "Kilaka": ["woman", "lover"],
    "Katakamukha": ["woman", "dancer"],

    # nature / soft
    "Pataka": ["person", "traveler", "figure", "wanderer"],
    "Hamsasya": ["graceful person"],
}


def pick(lst):
    return random.choice(lst)


def get_character(rasas, mudras):

    # 1. Try mudra-based role
    role_candidates = []

    for m in mudras:
        if m in mudra_role_map:
            role_candidates.extend(mudra_role_map[m])

    role_priority = {
        "king": 3,
        "warrior": 3,
        "devotee": 2,
        "woman": 1,
        "dancer": 1,
        "lover": 1,
        "traveler": 1
    }

    if role_candidates:
        role_scores = {}

        for r in role_candidates:
            role_scores[r] = role_scores.get(r, 0) + role_priority.get(r, 0)

        best_role = max(role_scores, key=role_scores.get)
        return f"The {best_role}"

        # 2. fallback to rasa (your original logic)
        dominant = rasas[0] if rasas else "Shanta"
        options = rasa_characters.get(dominant, ["The dancer"])
        return random.choice(options)

# ─────────────────────────────────────────────────────────────
#  UTILITIES (ROBUST)
# ─────────────────────────────────────────────────────────────


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def safe_pick(lst, default=""):
    return random.choice(lst) if lst else default


def clean_phrase(text):
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def unique_extend(base_list, new_items):
    for item in new_items:
        if item not in base_list:
            base_list.append(item)
    return base_list
# ─────────────────────────────────────────────────────────────
#  GESTURE SENTENCE — built from your description + meanings
# ─────────────────────────────────────────────────────────────


def gesture_sentence(character, mudra_name):
    if mudra_name not in mudra_info:
        return f"{character} forms an ancient gesture."

    info = mudra_info[mudra_name]

    desc = clean_phrase(info.get("description", "performs a gesture"))
    meanings = [clean_phrase(m) for m in info.get("meanings", []) if m]
    viniyoga = [clean_phrase(v) for v in info.get("viniyoga", []) if v]

    meaning_word = safe_pick(meanings, "expression")
    usage = safe_pick(viniyoga, "")

    # avoid ugly duplication like: "gesture of love, love"
    if usage and meaning_word in usage:
        return f"{character} {desc} — expressing {usage}."
    elif usage:
        return f"{character} {desc} — a gesture of {meaning_word}, often used for {usage}."
    else:
        return f"{character} {desc} — expressing {meaning_word}."

# ─────────────────────────────────────────────────────────────
#  CORE FUNCTION
# ─────────────────────────────────────────────────────────────

def mudras_and_rasas_to_story(mudras: list, rasas: list) -> str:
    if not mudras and not rasas:
        return "No gestures or expressions provided."

    valid_mudras = [m for m in mudras if m in mudra_info]
    valid_rasas = [r for r in rasas if r in rasa_story]

    if not valid_rasas:
        valid_rasas = ["Shanta"]

    # ✅ STEP 1: get dominant rasa FIRST
    dominant_rasa = valid_rasas[0]
    vocab = rasa_story[dominant_rasa]

    # ✅ STEP 2: build tone correctly
    if len(valid_rasas) > 1:
        tone = pick(vocab["tones"]) + " yet " + pick(rasa_story[valid_rasas[1]]["tones"])
    else:
        tone = pick(vocab["tones"])

    # ✅ STEP 3: character
    character = get_character(valid_rasas, valid_mudras)

    # ✅ sentence 1
    s1 = f"{character} stands {tone} in {pick(vocab['settings'])}."

    # ✅ gesture sentences
    gesture_sentences = []
    for mudra in valid_mudras[:3]:
        gesture_sentences.append(gesture_sentence(character, mudra))

    # ✅ fallback if no mudras
    if not gesture_sentences:
        gesture_sentences.append(
            f"{character} {pick(vocab['actions'])} {tone}, "
            f"moved by deep {pick(vocab['emotion'])}."
        )

    # ✅ climax
    climax = (
        f"{character} {pick(vocab['actions'])} {tone}, "
        f"overcome with {pick(vocab['emotion'])}."
    )

    # ✅ closing
    closing = vocab["closing"]

    return " ".join([s1] + gesture_sentences + [climax, closing])
# ─────────────────────────────────────────────────────────────
#  EXAMPLES
# ─────────────────────────────────────────────────────────────


examples = [
    {"label": "Devotee at the temple",              "mudras": [
        "Anjali", "Pushpaputa"],     "rasas": ["Shanta"]},
    {"label": "Warrior on the battlefield",         "mudras": [
        "Mushti", "Trishula"],       "rasas": ["Veera", "Raudra"]},
    {"label": "Frightened traveler in dark forest", "mudras": [
        "Pataka"],                   "rasas": ["Bhayanaka"]},
    {"label": "Sage beholding the crescent moon",   "mudras": [
        "Chandrakala", "Alapadma"],  "rasas": ["Adbhuta"]},
    {"label": "Woman in grief",                     "mudras": [
        "Utsanga"],                  "rasas": ["Karuna"]},
    {"label": "Dancer in love",                     "mudras": [
        "Katakamukha", "Kilaka"],    "rasas": ["Shringara"]},
    {"label": "Child celebrating with joy",         "mudras": [
        "Alapadma"],                 "rasas": ["Hasya"]},
    {"label": "Warrior reacting with disgust",      "mudras": [
        "Sarpashirsha"],             "rasas": ["Bibhatsa"]},
    {"label": "King in fury",                       "mudras": [
        "Mushti", "Gajadanta"],      "rasas": ["Raudra"]},
]

examples.extend([

    # ───────────────── BASIC EDGE CASES ─────────────────

    {"label": "No mudras, only rasa",
     "mudras": [],
     "rasas": ["Shanta"]},

    {"label": "No rasas, only mudras",
     "mudras": ["Anjali", "Pushpaputa"],
     "rasas": []},

    {"label": "Empty input",
     "mudras": [],
     "rasas": []},

    # ───────────────── ROLE CONFLICT TESTS ─────────────────

    {"label": "King + devotion mix",
     "mudras": ["Gajadanta", "Anjali"],
     "rasas": ["Shanta"]},

    {"label": "Warrior + love (conflict)",
     "mudras": ["Mushti", "Kilaka"],
     "rasas": ["Shringara"]},

    {"label": "Devotee + fear",
     "mudras": ["Anjali"],
     "rasas": ["Bhayanaka"]},

    # ───────────────── MULTI-MUDRA COMPLEX ─────────────────

    {"label": "Temple ritual sequence",
     "mudras": ["Anjali", "Pushpaputa", "Kapota"],
     "rasas": ["Shanta"]},

    {"label": "Battle escalation",
     "mudras": ["Mushti", "Trishula", "Kartarimukha"],
     "rasas": ["Veera", "Raudra"]},

    {"label": "Love expression sequence",
     "mudras": ["Alapadma", "Katakamukha", "Kilaka"],
     "rasas": ["Shringara"]},

    # ───────────────── ENVIRONMENT + RASA STRESS ─────────────────

    {"label": "Fear + wonder mix",
     "mudras": ["Chandrakala"],
     "rasas": ["Bhayanaka", "Adbhuta"]},

    {"label": "Disgust + anger mix",
     "mudras": ["Sarpashirsha", "Mushti"],
     "rasas": ["Bibhatsa", "Raudra"]},

    # ───────────────── UNKNOWN / NOISY INPUT ─────────────────

    {"label": "Unknown mudra included",
     "mudras": ["Anjali", "FakeMudra"],
     "rasas": ["Shanta"]},

    {"label": "Unknown rasa included",
     "mudras": ["Mushti"],
     "rasas": ["UnknownRasa"]},

    # ───────────────── SINGLE STRONG SIGNALS ─────────────────

    {"label": "Pure anger",
     "mudras": ["Mushti"],
     "rasas": ["Raudra"]},

    {"label": "Pure peace",
     "mudras": ["Anjali"],
     "rasas": ["Shanta"]},

    {"label": "Pure wonder",
     "mudras": ["Chandrakala"],
     "rasas": ["Adbhuta"]},

    # ───────────────── LONGER INPUT STRESS ─────────────────

    {"label": "Complex mixed narrative",
     "mudras": ["Anjali", "Alapadma", "Mushti", "Trishula"],
     "rasas": ["Shringara", "Veera", "Raudra"]},

    {"label": "Too many mudras (limit test)",
     "mudras": ["Anjali", "Pushpaputa", "Kapota", "Mushti", "Trishula", "Alapadma"],
     "rasas": ["Shanta"]},

])
# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 66)
    print("  REVERSE PIPELINE — Mudras + Rasas → Story")
    print("=" * 66)

    print("\n── Built-in Examples ──────────────────────────────────────────\n")
    for ex in examples:
        print(f"[ {ex['label']} ]")
        print(f"  Mudras : {', '.join(ex['mudras'])}")
        print(f"  Rasas  : {', '.join(ex['rasas'])}")
        print(
            f"  Story  : {mudras_and_rasas_to_story(ex['mudras'], ex['rasas'])}")
        print()

    print("── Interactive Mode ────────────────────────────────────────────")
    print("  Available Mudras :", ", ".join(sorted(mudra_info.keys())))
    print("  Available Rasas  :", ", ".join(sorted(rasa_story.keys())))
    print()

    while True:
        mudra_input = input(
            "Enter mudras (comma-separated, or 'exit'): ").strip()
        if mudra_input.lower() == "exit":
            break
        rasa_input = input("Enter rasas  (comma-separated): ").strip()
        if rasa_input.lower() == "exit":
            break

        mudras = [m.strip() for m in mudra_input.split(",") if m.strip()]
        rasas = [r.strip() for r in rasa_input.split(",") if r.strip()]

        bad_mudras = [m for m in mudras if m not in mudra_info]
        bad_rasas = [r for r in rasas if r not in rasa_story]
        if bad_mudras:
            print(f"  Warning — unrecognized mudras: {bad_mudras}")
        if bad_rasas:
            print(f"  Warning — unrecognized rasas: {bad_rasas}")

        print(f"\n  Story:\n  {mudras_and_rasas_to_story(mudras, rasas)}\n")

    print("\nProgram finished.")
