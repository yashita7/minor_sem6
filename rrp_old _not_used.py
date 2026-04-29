import pandas as pd
import ast
import re
import random
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.data.path.append("./nltk_data")

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir='./nltk_data')

lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────────
#  LOAD DATASETS
# ─────────────────────────────────────────────

mudra_df = pd.read_csv("./data/mudras.csv")
face_df  = pd.read_csv("./data/facial_expressions.csv")

mudra_df["meanings"] = mudra_df["meanings"].apply(ast.literal_eval)
mudra_df["viniyoga"] = mudra_df["viniyoga"].apply(ast.literal_eval)
face_df["meanings"]  = face_df["meanings"].apply(ast.literal_eval)
face_df["viniyoga"]  = face_df["viniyoga"].apply(ast.literal_eval)

# ─────────────────────────────────────────────
#  BUILD REVERSE LOOKUP TABLES
#  mudra name  → { meanings, viniyoga }
#  rasa name   → { meanings, viniyoga }
# ─────────────────────────────────────────────

mudra_info = {}
for _, row in mudra_df.iterrows():
    mudra_info[row["transliteration"]] = {
        "meanings": row["meanings"],
        "viniyoga": row["viniyoga"],
        "english":  row.get("english_name", row["transliteration"])
    }

rasa_info = {}
for _, row in face_df.iterrows():
    rasa_info[row["transliteration"]] = {
        "meanings": row["meanings"],
        "viniyoga": row["viniyoga"],
        "english":  row.get("english_name", row["transliteration"])
    }

# ─────────────────────────────────────────────
#  RASA → STORY VOCABULARY
#  Each rasa carries: tone words, setting words,
#  action verbs, emotional descriptors
# ─────────────────────────────────────────────

rasa_vocab = {
    "Shringara": {
        "tone":    ["gently", "softly", "tenderly", "with longing", "with grace"],
        "setting": ["the blooming garden", "the moonlit river bank", "the quiet grove of jasmine",
                    "the lotus pond at dawn", "the courtyard filled with fragrance"],
        "actions": ["gazes", "adorns", "offers", "moves", "dances", "reaches out"],
        "emotion": ["love", "beauty", "longing", "devotion", "tenderness"],
        "mood":    "The air is filled with sweetness."
    },
    "Hasya": {
        "tone":    ["playfully", "with laughter", "joyfully", "with delight", "brightly"],
        "setting": ["the sunlit meadow", "the busy festival ground", "the riverside",
                    "the open courtyard", "the village square"],
        "actions": ["dances", "leaps", "spins", "claps", "runs", "plays"],
        "emotion": ["joy", "laughter", "delight", "celebration", "happiness"],
        "mood":    "The world feels light and bright."
    },
    "Karuna": {
        "tone":    ["slowly", "with heavy heart", "in silence", "sorrowfully", "with tears"],
        "setting": ["the empty riverbank", "the barren field", "the grey evening sky",
                    "the silent forest path", "the darkened doorway"],
        "actions": ["weeps", "kneels", "waits", "searches", "remembers", "mourns"],
        "emotion": ["grief", "sorrow", "loss", "longing", "compassion"],
        "mood":    "A deep sadness hangs in the air."
    },
    "Raudra": {
        "tone":    ["fiercely", "with burning rage", "with fury", "violently", "with great force"],
        "setting": ["the battlefield", "the burning ground", "the stormy sky",
                    "the crumbling fortress", "the dusty arena"],
        "actions": ["strikes", "roars", "charges", "destroys", "confronts", "challenges"],
        "emotion": ["anger", "fury", "wrath", "rage", "vengeance"],
        "mood":    "The earth trembles beneath the weight of wrath."
    },
    "Veera": {
        "tone":    ["boldly", "with great strength", "fearlessly", "with determination", "with pride"],
        "setting": ["the grand battlefield", "the mountain peak", "the royal hall",
                    "the open plain under a vast sky", "the gate of the fortress"],
        "actions": ["stands", "raises", "marches", "commands", "defeats", "protects"],
        "emotion": ["courage", "heroism", "strength", "honor", "determination"],
        "mood":    "Victory is written in every step."
    },
    "Bhayanaka": {
        "tone":    ["trembling", "with fear", "in terror", "with wide eyes", "breathlessly"],
        "setting": ["the dark forest at night", "the foggy path", "the haunted ruins",
                    "the riverbank in the dead of night", "the shadowed cave"],
        "actions": ["freezes", "hides", "runs", "trembles", "watches", "backs away"],
        "emotion": ["fear", "terror", "dread", "panic", "horror"],
        "mood":    "Every shadow holds a threat."
    },
    "Bibhatsa": {
        "tone":    ["with revulsion", "turning away", "in disgust", "with horror", "slowly"],
        "setting": ["the forsaken place", "the rotting earth", "the abandoned grounds",
                    "the filthy alley", "the ruined shrine"],
        "actions": ["recoils", "turns away", "covers face", "steps back", "refuses"],
        "emotion": ["disgust", "repulsion", "rejection", "aversion", "horror"],
        "mood":    "The sight is unbearable."
    },
    "Adbhuta": {
        "tone":    ["in awe", "with wide eyes", "in wonder", "breathlessly", "in amazement"],
        "setting": ["beneath the vast night sky", "at the edge of the horizon", "beside the glowing river",
                    "on the mountaintop", "in the divine light of the moon"],
        "actions": ["stares", "marvels", "stands still", "gazes upward", "witnesses"],
        "emotion": ["wonder", "amazement", "awe", "astonishment", "reverence"],
        "mood":    "The world reveals something extraordinary."
    },
    "Shanta": {
        "tone":    ["quietly", "in stillness", "peacefully", "with deep calm", "with devotion"],
        "setting": ["the banks of the river", "the quiet temple courtyard", "the forest clearing",
                    "the still pond at sunrise", "the garden of meditation"],
        "actions": ["sits", "closes eyes", "breathes", "prays", "offers", "meditates"],
        "emotion": ["peace", "calm", "serenity", "devotion", "surrender"],
        "mood":    "Everything is still. Everything is enough."
    }
}

# Mudra → what gesture looks like / what it expresses (short phrase for story use)
mudra_gesture_phrases = {
    "Anjali":      ["raises both hands together in prayer",
                    "joins palms gently before the heart",
                    "brings hands together in a gesture of offering"],
    "Alapadma":    ["opens one hand like a blooming lotus",
                    "spreads fingers wide like petals unfolding",
                    "lifts a hand in the gesture of the full-blown flower"],
    "Chandrakala": ["raises a hand shaped like the crescent moon",
                    "holds a hand curved delicately like the new moon",
                    "forms the crescent with slow grace"],
    "Mushti":      ["closes the hand into a firm fist",
                    "tightens the fist with fierce resolve",
                    "raises a clenched fist toward the sky"],
    "Pataka":      ["extends one hand flat like a banner",
                    "sweeps an open hand through the air",
                    "holds the hand level like a flag in wind"],
    "Tripataka":   ["bends the ring finger down with purpose",
                    "forms the three-part gesture with steady grace"],
    "Kartarimukha":["separates two fingers like a pair of scissors",
                    "points two fingers apart in a sharp gesture"],
    "Mayura":      ["curls fingers into the shape of a peacock beak",
                    "brings thumb and forefinger together like a peacock's crest"],
    "Ardhachandra":["curves the hand like a half moon",
                    "opens the hand in the shape of the crescent"],
    "Arala":       ["bends the index finger inward in a curved gesture",
                    "holds a hand with one finger curved like a vine"],
    "Shukatunda":  ["extends one finger forward with intention",
                    "points with a single extended finger"],
    "Mushtishankha":["wraps one hand around the other in the conch gesture",
                    "forms the conch shell with both hands"],
    "Shivalinga":  ["forms the linga shape with both hands in reverence",
                    "holds one hand upright over the other in devotion"],
    "Trishula":    ["raises three fingers upward like the divine trident",
                    "points three fingers to the sky"],
    "Pushpaputa":  ["cups both hands gently as if holding petals",
                    "cradles open hands together like a flower offering"],
    "Samputa":     ["cups both hands together forming a hollow",
                    "brings hands together like a sealed vessel"],
    "Katakamukha": ["pinches thumb and forefinger in a delicate curve",
                    "forms the bracelet gesture with quiet elegance"],
    "Gajadanta":   ["bends the arm outward like an elephant's tusk",
                    "holds the arm curved and strong like a tusk"],
    "Pasha":       ["interlocks fingers like the loops of a snare",
                    "entangles the fingers in the noose gesture"],
    "Kilaka":      ["links two fingers together like a hook",
                    "interlocks forefingers in the chain gesture"],
    "Utsanga":     ["crosses the arms gently across the chest",
                    "wraps arms around oneself in an embrace"],
}

# Character templates — single entities only
character_templates = {
    "warrior":  ["The warrior", "The great warrior", "The lone warrior"],
    "devotee":  ["The devotee", "The young devotee", "The humble devotee"],
    "dancer":   ["The dancer", "The celestial dancer", "The lone dancer"],
    "sage":     ["The sage", "The old sage", "The forest sage"],
    "child":    ["The child", "The young child", "The small child"],
    "king":     ["The king", "The great king", "The proud king"],
    "woman":    ["The woman", "The young woman", "She"],
    "man":      ["The man", "He", "The lone figure"],
}

# Map rasas to likely character types
rasa_to_character = {
    "Raudra":    ["warrior", "king"],
    "Veera":     ["warrior", "king"],
    "Bhayanaka": ["devotee", "child", "man", "woman"],
    "Shanta":    ["devotee", "sage"],
    "Shringara": ["woman", "dancer"],
    "Hasya":     ["child", "dancer"],
    "Karuna":    ["woman", "man", "devotee"],
    "Adbhuta":   ["sage", "devotee", "dancer"],
    "Bibhatsa":  ["man", "woman", "warrior"],
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def pick(lst):
    return random.choice(lst)

def get_mudra_phrase(mudra):
    if mudra in mudra_gesture_phrases:
        return pick(mudra_gesture_phrases[mudra])
    # fallback: generic
    return f"forms the {mudra} gesture"

def get_meanings(mudra_name):
    if mudra_name in mudra_info:
        return mudra_info[mudra_name]["meanings"]
    return []

def choose_character(rasas):
    # pick based on dominant rasa
    for rasa in rasas:
        if rasa in rasa_to_character:
            ctype = pick(rasa_to_character[rasa])
            return pick(character_templates[ctype])
    return "The dancer"

def collect_keywords(mudras, rasas):
    """Gather meaningful words from mudra meanings + rasa vocabulary."""
    keywords = set()
    for m in mudras:
        for word in get_meanings(m):
            keywords.add(word.lower())
    for r in rasas:
        if r in rasa_vocab:
            keywords.update(rasa_vocab[r]["emotion"])
    return keywords

# ─────────────────────────────────────────────
#  CORE: BUILD STORY FROM MUDRAS + RASAS
# ─────────────────────────────────────────────

def mudras_and_rasas_to_story(mudras: list, rasas: list) -> str:
    """
    Takes a list of mudra names and rasa names.
    Returns a short single-entity story (3–5 sentences).
    """

    if not mudras and not rasas:
        return "No gestures or expressions provided."

    # 1. Pick dominant rasa (first in list = most important)
    dominant_rasa = rasas[0] if rasas else "Shanta"
    vocab = rasa_vocab.get(dominant_rasa, rasa_vocab["Shanta"])

    # 2. Pick character
    character = choose_character(rasas)

    # 3. Build setting sentence
    setting = pick(vocab["setting"])
    setting_sentence = f"{character} stands {pick(vocab['tone'])} in {setting}."

    # 4. Build gesture sentences (one per mudra, max 3)
    gesture_sentences = []
    for mudra in mudras[:3]:
        phrase = get_mudra_phrase(mudra)
        meanings = get_meanings(mudra)
        # pick a meaning word to weave into the sentence
        meaning_word = pick(meanings).lower() if meanings else ""
        action = pick(vocab["actions"])

        if meaning_word:
            gs = f"{character.split()[0] if character.split() else character} {phrase}, expressing {meaning_word}."
        else:
            gs = f"{character.split()[0] if character.split() else character} {phrase}."
        gesture_sentences.append(gs)

    # 5. Emotion climax sentence
    emotion_word = pick(vocab["emotion"])
    tone_word    = pick(vocab["tone"])
    action_word  = pick(vocab["actions"])
    climax = f"{character.split()[0] if character.split() else character} {action_word} {tone_word}, overcome with {emotion_word}."

    # 6. Mood closing line
    closing = vocab["mood"]

    # 7. Assemble — combine smoothly
    story_parts = [setting_sentence] + gesture_sentences + [climax, closing]
    story = " ".join(story_parts)

    return story


# ─────────────────────────────────────────────
#  VERBOSE VERSION: shows reasoning
# ─────────────────────────────────────────────

def mudras_and_rasas_to_story_verbose(mudras: list, rasas: list) -> dict:
    """
    Same as above but returns a dict with intermediate steps shown.
    """
    result = {}
    result["input_mudras"] = mudras
    result["input_rasas"]  = rasas

    dominant_rasa = rasas[0] if rasas else "Shanta"
    result["dominant_rasa"] = dominant_rasa

    character = choose_character(rasas)
    result["character"] = character

    mudra_meanings_used = {m: get_meanings(m) for m in mudras}
    result["mudra_meanings"] = mudra_meanings_used

    story = mudras_and_rasas_to_story(mudras, rasas)
    result["story"] = story

    return result


# ─────────────────────────────────────────────
#  EXAMPLES
# ─────────────────────────────────────────────

examples = [
    {
        "label": "Devotee at the temple",
        "mudras": ["Anjali", "Pushpaputa"],
        "rasas":  ["Shanta", "Shringara"]
    },
    {
        "label": "Warrior on the battlefield",
        "mudras": ["Mushti", "Trishula"],
        "rasas":  ["Veera", "Raudra"]
    },
    {
        "label": "Frightened traveler in the dark forest",
        "mudras": ["Pataka"],
        "rasas":  ["Bhayanaka"]
    },
    {
        "label": "Child in wonder under the night sky",
        "mudras": ["Alapadma", "Chandrakala"],
        "rasas":  ["Adbhuta"]
    },
    {
        "label": "Woman in grief and longing",
        "mudras": ["Utsanga"],
        "rasas":  ["Karuna", "Shringara"]
    },
    {
        "label": "Sage in deep meditation",
        "mudras": ["Shivalinga", "Anjali"],
        "rasas":  ["Shanta"]
    },
    {
        "label": "King in pride and anger",
        "mudras": ["Gajadanta", "Mushti"],
        "rasas":  ["Raudra", "Veera"]
    },
    {
        "label": "Dancer celebrating with joy",
        "mudras": ["Katakamukha", "Alapadma"],
        "rasas":  ["Hasya", "Shringara"]
    },
]


# ─────────────────────────────────────────────
#  MAIN: RUN EXAMPLES + INTERACTIVE MODE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 66)
    print("  REVERSE PIPELINE — Mudras + Rasas → Story")
    print("=" * 66)

    print("\n── Built-in Examples ──────────────────────────────────────────\n")

    for ex in examples:
        print(f"[ {ex['label']} ]")
        print(f"  Mudras : {', '.join(ex['mudras'])}")
        print(f"  Rasas  : {', '.join(ex['rasas'])}")
        story = mudras_and_rasas_to_story(ex["mudras"], ex["rasas"])
        print(f"  Story  : {story}")
        print()

    print("── Interactive Mode ────────────────────────────────────────────")
    print("  Enter mudra names and rasa names to generate a story.")
    print("  Type 'exit' to quit.\n")

    # show available names
    print("  Available Mudras :", ", ".join(sorted(mudra_info.keys())))
    print("  Available Rasas  :", ", ".join(sorted(rasa_info.keys())))
    print()

    while True:
        mudra_input = input("Enter mudras (comma-separated, or 'exit'): ").strip()
        if mudra_input.lower() == "exit":
            break

        rasa_input = input("Enter rasas  (comma-separated): ").strip()
        if rasa_input.lower() == "exit":
            break

        mudras = [m.strip() for m in mudra_input.split(",") if m.strip()]
        rasas  = [r.strip() for r in rasa_input.split(",")  if r.strip()]

        # validate
        bad_mudras = [m for m in mudras if m not in mudra_info and m not in mudra_gesture_phrases]
        bad_rasas  = [r for r in rasas  if r not in rasa_vocab]

        if bad_mudras:
            print(f"  Warning: unrecognized mudras: {bad_mudras}")
        if bad_rasas:
            print(f"  Warning: unrecognized rasas: {bad_rasas}")

        story = mudras_and_rasas_to_story(mudras, rasas)
        print(f"\n  Generated Story:\n  {story}\n")

    print("\nProgram finished.")