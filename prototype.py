"""
Streamlit web application for my MS Thesis Experiment.

This app simulates a health chatbot (HopDoc) with different conditions
to test user perceptions based on avatar presence and linguistic style matching (LSM).
It uses the Google Gemini API for generating responses and includes an enhanced
LSM simulation based on detected user style traits.

Major Changes:
- Added Participant ID input screen.
- Implemented comprehensive JSONL logging for experiment data.
- Added session timeout logic.
- Switched tokenization to NLTK's word_tokenize for potentially better handling.
- Refined post-processing to remove double spaces.
- Removed unused imports.
"""

import os
import re
import time
import json
from datetime import datetime, timezone
import streamlit as st
import emoji
import functools
import google.generativeai as genai
import nltk
import textstat
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from empath import Empath
from nltk.tokenize import word_tokenize

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide")

# Initialize Empath lexicon
lexicon = Empath()

# --- NLTK / VADER Setup ---
NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(NLTK_DIR):
    os.makedirs(NLTK_DIR)
nltk.data.path.append(NLTK_DIR)

# Function to download NLTK data safely
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt', download_dir=NLTK_DIR)
        st.success("NLTK 'punkt' downloaded.")

    try:
        nltk.data.find('sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt')
    except LookupError:
        st.info("Downloading NLTK 'vader_lexicon' data...")
        nltk.download('vader_lexicon', download_dir=NLTK_DIR)
        st.success("NLTK 'vader_lexicon' downloaded.")

download_nltk_data()
sia = SentimentIntensityAnalyzer()

# --- Logging Setup ---
LOG_DIR = "experiment_logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def log_event(event_data):
    """Logs an event dictionary to a participant-specific JSONL file."""
    if 'participant_id' not in st.session_state or not st.session_state.participant_id:
        print("Warning: Attempted to log event without participant ID.")
        return # Don't log if participant ID isn't set

    log_file = st.session_state.get("log_file_path")
    if not log_file:
        print("Error: Log file path not set in session state.")
        return

    try:
        # Add timestamp and participant ID to every log entry
        event_data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        event_data["participant_id"] = st.session_state.participant_id
        event_data["session_id"] = st.session_state.session_id # Unique ID for this specific run

        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(event_data, f, ensure_ascii=False)
            f.write('\n')
    except Exception as e:
        st.error(f"Logging Error: Failed to write to log file {log_file}. Error: {e}")
        print(f"Logging Error: Failed to write event {event_data} to {log_file}. Error: {e}")


# --- Pre-compile regex patterns & define function words ---
INFORMAL_RE = re.compile(
    # Original + Previous additions:
    r"\b(sup|bruh|yo|dude|frank[iy]+|lol|lmao|deadass|ain't|gonna|wanna|gotta|"
    r"kinda|sorta|lemme|dunno|ya|nah|tho|fr|btw|imo|idk|omg|ur|"
    r"u|yea+h*|funn+y+|ben|gotta|here+|playa|"
    # Acronyms:
    r"smh|tbh|omfg|wtf|idc|fyi|jk|btw|rn|lmk|"
    # Leet/Numbers:
    r"gr[38]t|l8r|b4|sk8|h8)\b|" # Added common leet speak
    # Laughter:
    r"\b(ha){2,}\b|\b(he){2,}\b|\b(hi){2,}\b|"
    # Elongated Vowels (Keep, but test):
    r"\b\w*([aeiou])\1{2,}\w*\b|"
    # Repeated Punctuation (end of string or followed by space):
    r"[!?\.]{2,}(\s|$)|" # Match two or more !, ?, .
    # Basic Emoticons / Kaomoji (add more if needed):
    r":\-?p\b|:\-?3\b|;\-?p\b|:\-?d\b|<3|:\'\((?!\w)|¯\\_\(ツ\)_/¯", # Added :'( , shrug
    re.IGNORECASE | re.UNICODE # Add UNICODE flag for shrug etc.
)
HEDGING_RE = re.compile(
    r"\b(maybe|not sure|kinda|sort of|might|possibly|perhaps|seems|appears|"
    r"suggests|i think|i guess|idk)\b",
    re.IGNORECASE
)
# Add new categories
# Note: This is a basic list, can be expanded. Be mindful of overlaps if any.
AUX_VERBS = {"am", "is", "are", "was", "were", "be", "being", "been",
             "have", "has", "had", "having",
             "do", "does", "did",
             "will", "would", "shall", "should", "may", "might", "must", "can", "could"}

CONJUNCTIONS = {"and", "but", "or", "so", "yet", "for", "nor", # Coordinating
                "while", "whereas", "although", "though", "because", "since", "if", "unless", "until", "when", "as", "that", "whether", "after", "before"} # Subordinating (common ones)

NEGATIONS = {"no", "not", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "ain't", "don't", "isn't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "mightn't", "mustn't", "can't", "couldn't"}
# Note: NLTK tokenization splits contractions like "don't" -> "do", "n't". Need to adjust tokenization or list.
# Let's adjust tokenization to keep contractions for negation matching.

FUNCTION_WORDS = {
    "i","you","we","he","she","they","me","him","her","us","them",
    "a","an","the",
    "in","on","at","by","for","with","about","against","between",
    "into","through","during","before","after","above","below","to",
    "from","up","down","over","under"
} | AUX_VERBS | CONJUNCTIONS | NEGATIONS # Combine all for general function word ratio


# Updated LSM_CATEGORIES dictionary
LSM_CATEGORIES = {
    "pronouns": {"i","you","we","he","she","they","me","him","her","us","them"},
    "articles":  {"a","an","the"},
    "prepositions": {
        "in","on","at","by","for","with","about","against","between",
        "into","through","during","before","after","above","below",
        "to","from","up","down","over","under"
    },
    "aux_verbs": AUX_VERBS,
    "conjunctions": CONJUNCTIONS,
    "negations": NEGATIONS,
    # Add quantifiers if desired later
}

# --- Helper Functions (Updated Tokenization for Negations/LSM Counts) ---
# Define the pattern for valid tokens including n't
valid_token_pattern = re.compile(r"^[a-z0-9]+$|^n't$", re.IGNORECASE)

def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes text using NLTK, preserving contractions (like n't) via regex filter,
    and returns lowercase tokens.
    """
    try:
        tokens = word_tokenize(text.lower())
        # Use the regex pattern to filter tokens
        return [t for t in tokens if valid_token_pattern.match(t)]
    except Exception as e:
        print(f"Tokenization error: {e}. Falling back to simple split.")
        # Fallback regex updated to try and capture n't
        return re.findall(r"\w+|n't", text.lower())

# Updated get_category_fraction to be simpler with correct tokenization
def get_category_fraction(text: str, word_set: set) -> float:
    tokens = tokenize_text(text) # Correct tokenization now keeps/removes as needed
    if not tokens:
        return 0.0
    # Now that tokenization is correct, simple check is enough
    count = sum(1 for t in tokens if t in word_set)
    return count / len(tokens)

# Also update detect_style_traits to use the corrected count logic if needed for LSM counts:
# compute_lsm_score remains structurally the same, but will now iterate over more categories
def compute_lsm_score(user_text: str, bot_text: str) -> float:
    scores = []
    for cat_words in LSM_CATEGORIES.values(): # Iterates over all categories now
        fu = get_category_fraction(user_text, cat_words)
        fb = get_category_fraction(bot_text,  cat_words)
        denom = fu + fb
        if denom > 1e-6:
             scores.append(1 - abs(fu - fb) / denom)
        elif abs(fu - fb) < 1e-6:
             scores.append(1.0)
        else:
             scores.append(0.0)

    return sum(scores) / len(scores) if scores else 1.0


# FINAL RECOMMENDED VERSION for post_process_response
def post_process_response(resp: str, is_adaptive: bool) -> str:
    """Cleans up bot response, removing informality/emojis for static style,
       and fixing common markdown list formatting (ensuring newline before items)."""
    if not is_adaptive:
        # Use emoji library to remove emojis
        resp = emoji.replace_emoji(resp, replace="")
        # Replace informal words with empty string
        resp = re.sub(INFORMAL_RE, "", resp) # Use updated INFORMAL_RE

    # Remove extra whitespace first
    resp = re.sub(r'\s{2,}', ' ', resp).strip()

    # --- Add Markdown List Fix ---
    # Ensure newline BEFORE list items (* or -) if not already there.
    # Looks for char NOT a newline, followed by optional newline,
    # then optional space, then bullet (* or -), then required space.
    # Replaces with the found char + \n + the space/bullet/space part.
    resp = re.sub(r"([^\n])(\n?)(\s*[\*-]\s+)", r"\1\n\3", resp)

    # Also handle case where list starts at the very beginning of the response
    resp = re.sub(r"^(\s*[\*-]\s+)", r"\n\1", resp)


    # Remove extra spaces potentially introduced AFTER the bullet fix
    resp = re.sub(r'\s{2,}', ' ', resp).strip()
    # Collapse >2 consecutive newlines into max 2 (for paragraph spacing)
    resp = re.sub(r'\n{3,}', '\n\n', resp)

    return resp


# --- Global Constants ---
AVATAR_IMAGE_PATH = "franko.png" # Make sure this file exists
DEFAULT_BOT_NAME = "HopDoc (Franko)"
NO_AVATAR_BOT_NAME = "HopDoc"
GEMINI_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21" # Use a standard model name unless specific version is required
SESSION_TIMEOUT_SECONDS = 600  # 10 minutes
LOG_SESSION_EVENTS = True # Control logging easily
MIN_LSM_TOKENS = 15 # Minimum tokens for user AND bot response to update smoothed LSM
LSM_SMOOTHING_ALPHA = 0.4 # Smoothing factor (0 = ignore new, 1 = use only new)

# --- Style Detection Function ---
# def is_informal_token(tok): # Define if using frequency heuristic
#     # Very rare words (zipf < 2.5) or OOV that aren't likely proper nouns/numbers
#     # Basic check: ignore capitalized words and pure numbers for this heuristic
#     if not tok.istitle() and not tok.isdigit():
#          # Check if not in standard dictionary AND rare
#          if tok not in EN_WORDS and zipf_frequency(tok, "en") < 2.5:
#              return True
#     return False

def detect_style_traits(user_input: str) -> dict:
    lower_input = user_input.lower()
    tokens = tokenize_text(user_input)
    word_count = len(tokens)

    # Calculate informality based on regex AND potentially other methods
    regex_informal = bool(INFORMAL_RE.search(user_input)) # Search raw input for punctuation/emojis
    # heuristic_informal = any(is_informal_token(t) for t in tokens) # Uncomment if using frequency heuristic
    # slangnet_informal = any(t in slangnet_set for t in tokens) # Uncomment if using SlangNet

    # Combine flags (adjust logic as needed)
    is_informal = regex_informal # or heuristic_informal or slangnet_informal

    traits = {
        "emoji": emoji.emoji_count(user_input) > 0,
        "informal": is_informal, # Use combined flag
        "hedging": bool(HEDGING_RE.search(lower_input)),
        "questioning": user_input.strip().endswith("?"),
        "exclamatory": user_input.count("!") > 0,
        "short": word_count <= 10,
        "question_count": user_input.count("?"),
        "exclamation_count": user_input.count("!"),
    }

    # --- Explicit Request Detection ---
    meta_request = None
    if re.search(r"\b(short|shorter|concise|brief|less text|too long)\b", lower_input):
        meta_request = "shorter"
    # Updated regex for 'longer' requests
    elif re.search(r"\b(more detail|long|longer|lengthy|elaborate)\b", lower_input):
        meta_request = "longer"
    elif re.search(r"\b(simple|simpler|easy|easier)\b", lower_input):
         meta_request = "simpler"
    traits["meta_request"] = meta_request

    # --- Sentiment Analysis (Adjust threshold later in prompt generation) ---
    sentiment = sia.polarity_scores(user_input)
    traits.update({
        "sentiment_neg": sentiment["neg"],
        "sentiment_neu": sentiment["neu"],
        "sentiment_pos": sentiment["pos"],
        "sentiment_compound": sentiment["compound"],
    })

    # --- Sentence splitting and complexity ---
    sentences = [s.strip() for s in re.split(r"[.!?]+", user_input) if s.strip()]
    sentence_count = max(1, len(sentences))
    avg_sentence_length = word_count / sentence_count # Uses token count

    # Avg word length calculation (less critical now, but uses filtered tokens)
    avg_word_length = sum(len(w) for w in tokens) / max(1, word_count)

    try:
        flesch_ease = textstat.flesch_reading_ease(user_input)
        fk_grade = textstat.flesch_kincaid_grade(user_input)
    except Exception as e:
        print(f"Textstat error: {e}")
        flesch_ease = -1
        fk_grade = -1

    # --- Function Word Ratio (based on combined FUNCTION_WORDS set) ---
    func_count = sum(1 for w in tokens if w in FUNCTION_WORDS) # Uses updated set
    function_word_ratio = func_count / max(1, word_count)

    traits.update({
        "avg_sentence_length":   avg_sentence_length,
        "avg_word_length":       avg_word_length, # Less reliable with this tokenization
        "flesch_reading_ease":   flesch_ease,
        "fk_grade":              fk_grade,
        "function_word_ratio":   function_word_ratio,
    })

    # --- Empath Analysis ---
    try:
        empath_cats = lexicon.analyze(user_input, normalize=True)
        if empath_cats is None: empath_cats = {}
    except Exception as e:
        print(f"Empath analysis error: {e}")
        empath_cats = {}

    traits.update({
        "empath_social":    empath_cats.get("social", 0.0),
        "empath_cognitive": empath_cats.get("cognitive_processes", 0.0),
        "empath_affect":    empath_cats.get("affect", 0.0),
    })

    # --- LSM Category Counts ---
    lsm_counts = {}
    tokens_for_lsm = tokenize_text(lower_input) # Tokenize once for all counts
    if tokens_for_lsm: # Avoid division by zero if tokenization fails
        for cat_name, cat_words in LSM_CATEGORIES.items():
            is_negation_set = "not" in cat_words
            count = 0
            for t in tokens_for_lsm:
                if t in cat_words:
                    count += 1
                elif is_negation_set and t == "n't" and any(neg.endswith("n't") for neg in cat_words):
                    count += 1
            lsm_counts[cat_name] = count
    else:
        lsm_counts = {cat_name: 0 for cat_name in LSM_CATEGORIES} # Default to zero counts

    traits["lsm_counts"] = lsm_counts

    # --- Pronoun specific flags (still useful for direct 'I'/'we' cues) ---
    traits["pronouns"] = {
        "i":   bool(re.search(r"\bi\b", lower_input)),
        "you": bool(re.search(r"\byou\b", lower_input)),
        "we":  bool(re.search(r"\bwe\b", lower_input)),
    }

    return traits


# --- Dynamic Prompt Generation (Updated Cues) ---
def generate_dynamic_prompt(base_prompt: str, style_profile: dict) -> str:
    style_cues = []

    # --- High Priority: Explicit User Requests ---
    meta_request = style_profile.get("meta_request")
    if meta_request:
        if meta_request == "shorter":
            style_cues.append("**USER REQUEST**: Respond concisely (e.g., 1-3 short sentences or bullet points). Acknowledge if you were too verbose previously.")
        elif meta_request == "longer":
            style_cues.append("**USER REQUEST**: Provide more detail or elaborate further.")
        elif meta_request == "simpler":
             style_cues.append("**USER REQUEST**: Explain in simpler terms, using easier vocabulary.")

    # --- Linguistic Style Matching (LSM) Cues ---
    lsm_prev = style_profile.get("lsm_score_prev", None)
    lsm_counts = style_profile.get("lsm_counts", {})
    lsm_info_parts = []
    if lsm_prev is not None:
        lsm_info_parts.append(f"Previous LSM score: {lsm_prev:.2f}")

    pron_c = lsm_counts.get('pronouns', 0)
    art_c = lsm_counts.get('articles', 0)
    prep_c = lsm_counts.get('prepositions', 0)
    aux_c = lsm_counts.get('aux_verbs', 0)
    conj_c = lsm_counts.get('conjunctions', 0)
    neg_c = lsm_counts.get('negations', 0)

    lsm_info_parts.append(f"User counts: {pron_c}pn, {art_c}art, {prep_c}prep, {aux_c}aux, {conj_c}conj, {neg_c}neg.")

    if lsm_info_parts:
         style_cues.append(f"LSM Info: {' '.join(lsm_info_parts)} Consider aligning your function word density.")
         if lsm_prev is not None:
             if lsm_prev < 0.4:
                 style_cues.append("-> Style differs significantly, try matching user function word counts more closely.")
             elif lsm_prev < 0.7:
                 style_cues.append("-> Moderate alignment, maintain or slightly adjust function word usage.")


    # --- Sentiment Cues (Adjusted Threshold) ---
    comp = style_profile.get("sentiment_compound", 0)
    # Lowered negative threshold, slightly increased positive threshold
    if comp >= 0.6:
        style_cues.append("User tone seems positive. Respond with an upbeat, encouraging style.")
    elif comp <= -0.2: # Lowered threshold
        style_cues.append("User tone seems negative/frustrated. Respond with extra empathy and clarity.")

    # --- Length & Complexity Cues ---
    asl = style_profile.get("avg_sentence_length", 15)
    # Providing target length might be more effective
    style_cues.append(f"User avg sentence length: {asl:.1f} words. Aim for similar brevity/detail in your sentences.")
    # Example of conditional cue (less direct):
    # if asl > 20:
    #     style_cues.append("User messages are detailed (long sentences). Respond informatively but try to be reasonably concise unless asked for detail.")
    # elif asl < 8:
    #     style_cues.append("User messages are brief (short sentences). Keep your responses concise and to the point.")

    fre = style_profile.get("flesch_reading_ease", 100)
    if fre < 60: # Harder to read
        style_cues.append(f"User language complexity (FRE {fre:.0f}) suggests simpler wording may be better. Use clear, accessible language (aim for FRE > 60).")

    # --- Punctuation Cues ---
    q_count = style_profile.get("question_count", 0)
    e_count = style_profile.get("exclamation_count", 0)
    if style_profile.get("questioning"): # Check the boolean flag still
         style_cues.append(f"User asked a question ({q_count} '?'). Ensure you address it directly. Ask for clarification if ambiguous.")
    if e_count > 0:
        style_cues.append(f"User used {e_count} '!'. Use exclamation points very sparingly yourself for matching enthusiasm if appropriate.")


    # --- Other Style Cues (Informality, Hedging, Pronouns, Empath) ---
    if style_profile.get("emoji"):
        style_cues.append("User uses emojis. Include 1-2 relevant emojis if appropriate.")
    if style_profile.get("informal"):
        style_cues.append("User uses informal language/slang (e.g., 'u', 'lol', 'yeaah'). Adopt a slightly more relaxed, conversational tone. Avoid being stiffly formal.")
    if style_profile.get("hedging"):
        style_cues.append("User uses hedging/tentative language (e.g., 'maybe', 'I think'). Mirror this slightly with cautious phrasing where appropriate (e.g., 'It might be...', 'Some find...').")

    pronouns = style_profile.get("pronouns", {})
    if pronouns.get("we"):
        style_cues.append("User uses 'we'. Frame suggestions collaboratively using 'we' sometimes.")
    elif pronouns.get("you"):
        style_cues.append("User uses 'you'. Frame advice directly using 'you'.")
    elif pronouns.get("i"):
        # Make this more specific if 'I' count is high?
        style_cues.append("User uses 'I'. Use empathetic 'I' statements where appropriate (e.g., 'I understand...').")


    cat_scores = {
        "social": style_profile.get("empath_social", 0),
        "cognitive": style_profile.get("empath_cognitive", 0),
        "affect": style_profile.get("empath_affect", 0)
    }
    top_cat = max(cat_scores, key=cat_scores.get) if any(v > 0 for v in cat_scores.values()) else None
    if top_cat and cat_scores[top_cat] > 0.1:
        if top_cat == "social":
            style_cues.append("User's language is social. Emphasize connection/shared experiences if relevant.")
        elif top_cat == "cognitive":
            style_cues.append("User's language is cognitive/analytical. Provide logical explanations or structured info.")
        elif top_cat == "affect":
            style_cues.append("User's language is affect-rich (emotional). Acknowledge feelings and respond with empathy.")

    # --- Construct Final Prompt ---
    if style_cues:
        instruction_header = "\n\n## IMPORTANT Style Adaptation Instructions (Follow Closely):\n"
        formatted_cues = "- " + "\n- ".join(style_cues)
        final_prompt = base_prompt + instruction_header + formatted_cues
        return final_prompt
    else:
         # This should ideally not happen with the number of cues we have
         print("Warning: No style cues generated for adaptive prompt.")
         return base_prompt + "\n\n## Style Adaptation Instructions:\n- Maintain a generally helpful and supportive tone." # Basic fallback


# --- Gemini API Call (with Logging) ---
def get_gemini_response(user_prompt, chat_history, is_adaptive, show_avatar, style_profile=None):
    persona_name = DEFAULT_BOT_NAME if show_avatar else NO_AVATAR_BOT_NAME
    error_message = None # To store potential errors for logging
    

    # --- Base Prompts ---
    # Refined prompts slightly for clarity
    base_adaptive_prompt = f"""
You are {persona_name}, a friendly AI wellness assistant focused on general health topics like fitness, nutrition, sleep, and stress management.
Your primary goal is to be helpful, supportive, and informative.
**CRITICAL:** Adapt your response style based *only* on the user's *last* message, following the specific 'Style Adaptation Instructions' provided below.
Keep responses relevant to health and wellness.
**Strictly avoid giving medical advice or diagnoses.** You are not a doctor. If asked for medical advice, politely state your limitations and suggest consulting a healthcare professional.
Example Disclaimer: "As an AI assistant, I can't provide medical advice. For diagnosis or treatment, please consult a healthcare provider."
**IMPORTANT:** Do not preface your response with reasoning steps like 'Thinking process:', 'Step 1:', etc. Provide only the final user-facing answer.
"""
    static_prompt = f"""
You are {persona_name}, a friendly AI wellness assistant focused on general health topics like fitness, nutrition, sleep, and stress management.
Your primary goal is to be helpful, supportive, and informative.
**CRITICAL:** Maintain a CONSISTENT communication style: Empathetic, clear, encouraging, and informative. Use standard, proper language with moderate sentence length (aim for 1-3 sentences per point).
**Do NOT adapt or mimic the user's writing style (e.g., slang, emojis, sentence length).** Maintain *your* defined style regardless of how the user writes.
Keep responses relevant to health and wellness.
**Strictly avoid giving medical advice or diagnoses.** You are not a doctor. If asked for medical advice, politely state your limitations and suggest consulting a healthcare professional.
Example Disclaimer: "As an AI assistant, I can't provide medical advice. For diagnosis or treatment, please consult a healthcare provider."
**IMPORTANT:** Do not preface your response with reasoning steps like 'Thinking process:', 'Step 1:', etc. Provide only the final user-facing answer.
"""

    # Determine system instruction and log it
    if is_adaptive and style_profile:
        system_instruction = generate_dynamic_prompt(base_adaptive_prompt, style_profile)
        prompt_type = "adaptive"
    else:
        system_instruction = static_prompt
        prompt_type = "static"

    if LOG_SESSION_EVENTS:
        log_event({
            "event_type": "llm_request_start",
            "prompt_type": prompt_type,
            "system_instruction": system_instruction,
            "history_length": len(chat_history)
        })

    # Format message history for Gemini API
    messages = []
    # Add historical messages (limit history length if needed)
    for msg in chat_history: # Make sure history has 'role' and 'content'
         role = "model" if msg["role"] == "assistant" else "user"
         messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    # Add the current user prompt
    messages.append({"role": "user", "parts": [{"text": user_prompt}]})


    processed_response = "Sorry, I encountered a problem generating a response. Please try again." # Default error response

    try:
        # Ensure API key is configured
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
             raise ValueError("GOOGLE_API_KEY not found in Streamlit secrets.")
        genai.configure(api_key=api_key)

        # Create the model instance with the system instruction
        model = genai.GenerativeModel(GEMINI_MODEL_NAME, system_instruction=system_instruction)

        # Make the API call
        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=['\n\n\n'], # Optional: stop generation on triple newline
            # max_output_tokens=512,    # Optional: limit response length
            temperature=0.7,        # Adjust creativity (0.0 = deterministic, 1.0 = creative)
            # top_p=0.9,              # Optional: nucleus sampling
            # top_k=40                # Optional: top-k sampling
        )

        safety_settings = { # Adjust safety settings if needed (be cautious)
            # genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }


        resp = model.generate_content(
            messages,
            generation_config=generation_config,
            safety_settings=safety_settings
            )

        # Extract text, handling potential lack of response or blocked content
        if resp.candidates:
            if resp.candidates[0].content and resp.candidates[0].content.parts:
                 raw_response = resp.candidates[0].content.parts[0].text
            else:
                 # Check finish reason for blocked content etc.
                 finish_reason = resp.candidates[0].finish_reason
                 raw_response = f"Response generation finished reason: {finish_reason}. No content available."
                 error_message = f"Gemini Finish Reason: {finish_reason}"
        else:
             # Check prompt feedback for blocking reasons
             block_reason = resp.prompt_feedback.block_reason if resp.prompt_feedback else "Unknown"
             raw_response = f"Response generation failed. Prompt Feedback: {block_reason}"
             error_message = f"Gemini Prompt Feedback: {block_reason}"


        # Post-process the raw response (applies static style rules if not adaptive)
        processed_response = post_process_response(raw_response, is_adaptive)

        if LOG_SESSION_EVENTS:
            log_event({
                "event_type": "llm_response_success",
                "raw_response": raw_response,
                "processed_response": processed_response,
                "finish_reason": str(resp.candidates[0].finish_reason) if resp.candidates else "N/A",
                "safety_ratings": str(resp.candidates[0].safety_ratings) if resp.candidates else "N/A",
                "token_count": resp.usage_metadata.total_token_count if resp.usage_metadata else None
            })

    except Exception as e:
        error_message = f"Error calling Gemini API: {e}"
        st.error(error_message) # Show error in UI
        print(error_message)   # Print detailed error to console/server logs
        # Log the error
        if LOG_SESSION_EVENTS:
            log_event({
                "event_type": "llm_error",
                "error_message": str(e),
                "user_prompt": user_prompt # Log context of error
            })
        # Use the default error message defined earlier
        processed_response = "Sorry, an error occurred while trying to reach the assistant. Please check the connection or try again later."


    return processed_response, system_instruction # Return system instruction for logging


# --- UI Functions ---

def display_participant_id_input():
    """Shows the participant ID input screen."""
    st.title("Welcome to the HopDoc Experiment")
    st.markdown("Before we begin, please enter the **Participant ID** you were assigned.")
    pid_input = st.text_input("Participant ID (e.g., 01, 15):", max_chars=4, key="pid_input")
    if st.button("Submit ID"):
        # Basic validation: check if it's numeric and maybe length
        if pid_input.isdigit() and 1 <= len(pid_input) <= 4: # Allow 1 to 4 digits for flexibility
            st.session_state.participant_id = pid_input.zfill(2) # Pad with zero if needed (e.g., 5 -> 05)
            st.session_state.participant_id_entered = True

            # Set up logging file path based on validated ID
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            st.session_state.session_id = f"session_{timestamp_str}" # Unique ID for this run
            log_filename = f"participant_{st.session_state.participant_id}_{st.session_state.session_id}.jsonl"
            st.session_state.log_file_path = os.path.join(LOG_DIR, log_filename)

            # Log session start
            if LOG_SESSION_EVENTS:
                 log_event({
                    "event_type": "session_start",
                    "initial_condition": st.session_state.experiment_condition,
                 })
            st.rerun()
        else:
            st.warning("Please enter a valid numeric Participant ID (1-4 digits).")

def display_welcome_disclaimer(show_avatar_in_welcome):
    """Multi-step welcome + disclaimer UI."""
    if 'disclaimer_step' not in st.session_state:
        st.session_state.disclaimer_step = 1

    col1, col2 = st.columns([0.8, 0.2], gap="medium")
    with col1:
        st.title("Welcome to HopDoc")
        step = st.session_state.disclaimer_step
        time_limit_minutes = int(SESSION_TIMEOUT_SECONDS / 60)

        if step == 1:
            st.markdown(
                f"Hi there! I'm **Franko**, your friendly AI wellness assistant! "
                f"{'*(Avatar display simulated)*' if not show_avatar_in_welcome else ''}\n\n"
                f"For this session (approx. **{time_limit_minutes} minutes**), feel free to ask me general questions about health and wellness topics like fitness, nutrition, sleep, or stress management."
            )
            if st.button("Continue", key="disclaimer_1"):
                st.session_state.disclaimer_step = 2
                st.rerun()

        elif step == 2:
            st.markdown(
                "**⚠️ Important Disclaimer:**\n\n"
                "I am an AI simulation for a research study. I am **not** a medical professional, and **cannot** provide medical advice, diagnosis, or emergency support. "
                "The information I provide is for general knowledge and informational purposes only, and does not constitute medical advice.\n\n"
                "Please consult with a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment."
                " If you think you may have a medical emergency, call your doctor or emergency services immediately."
            )
            if st.button("I Understand and Agree", key="disclaimer_2"):
                st.session_state.disclaimer_step = 3
                st.rerun()

        else: # Step 3
            st.markdown(
                "Great! We're ready to start.\n\n"
                "Remember, you can ask about things like:\n"
                "* 🤔 \"How can I create a better sleep routine?\"\n"
                "* 🍎 \"What are some healthy snack ideas?\"\n"
                "* 🧘 \"Suggest simple ways to manage daily stress.\"\n"
            )
            if st.button("Let's Begin Chatting!", key="disclaimer_3"):
                st.session_state.disclaimer_accepted = True
                st.session_state.start_time = time.time()
                st.session_state.messages = [] # Initialize message list
                st.session_state.session_timed_out = False # Ensure timeout flag is reset

                # Log acceptance and generate initial greeting
                if LOG_SESSION_EVENTS:
                    log_event({
                        "event_type": "disclaimer_accepted",
                        "condition": st.session_state.experiment_condition,
                    })

                # Generate initial greeting (no user input context yet)
                initial_style_profile = None # No user input yet
                initial_greeting, system_instr = get_gemini_response(
                    user_prompt="<System Initiated Conversation Start>", # Special marker
                    chat_history=[],
                    is_adaptive=st.session_state.experiment_condition['lsm'],
                    show_avatar=st.session_state.experiment_condition['avatar'],
                    style_profile=initial_style_profile
                )

                # Log the initial bot message
                if LOG_SESSION_EVENTS:
                    log_event({
                        "event_type": "bot_response",
                        "turn_number": 0, # Initial turn
                        "user_prompt": "<System Initiated Conversation Start>",
                        "style_profile": initial_style_profile, # Will be None
                        "lsm_score": None, # No user input to compare
                        "system_instruction_used": system_instr, # Log the prompt used
                        "bot_response_raw": initial_greeting, # Log raw before post-processing (though likely same here)
                        "bot_response_processed": initial_greeting,
                        "condition": st.session_state.experiment_condition
                    })

                # Add greeting to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": initial_greeting,
                    "small_avatar_used": AVATAR_IMAGE_PATH if st.session_state.experiment_condition['avatar'] else None
                 })

                st.rerun() # Rerun to display the chat interface

    with col2:
        # Show avatar during disclaimer steps if condition requires it
        if show_avatar_in_welcome and st.session_state.disclaimer_step <= 3:
            if os.path.exists(AVATAR_IMAGE_PATH):
                st.image(AVATAR_IMAGE_PATH, width=120)
            else:
                st.caption("(Avatar image not found)")


# --- Session State Initialization ---
# Check for essential states, initialize if missing
if 'participant_id_entered' not in st.session_state:
    st.session_state.participant_id_entered = False
if 'participant_id' not in st.session_state:
    st.session_state.participant_id = None
if 'log_file_path' not in st.session_state:
    st.session_state.log_file_path = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None # Will be set after PID entry
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False
if 'experiment_condition' not in st.session_state:
    # Default condition - this might be set externally in a real experiment setup
    st.session_state.experiment_condition = {'avatar': True, 'lsm': True}
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'lsm_score' not in st.session_state:
    st.session_state.lsm_score = None # Store the *last* calculated score
if 'session_timed_out' not in st.session_state:
    st.session_state.session_timed_out = False
if 'smoothed_lsm_score' not in st.session_state:
    st.session_state.smoothed_lsm_score = 0.5 # Initialize at neutral midpoint    

# --- Sidebar Controls (Consider disabling after start for experiment) ---
# For prototyping, keep enabled. For actual experiment, you might lock these.
sidebar_disabled = st.session_state.get('disclaimer_accepted', False) # Disable after chat starts?

st.sidebar.title("Experiment Setup")
st.sidebar.markdown("*(Note: For a real experiment, these settings would likely be fixed per participant.)*")

# Load current state for radio buttons
current_avatar_index = 0 if st.session_state.experiment_condition.get('avatar', True) else 1
current_lsm_index = 0 if st.session_state.experiment_condition.get('lsm', True) else 1

avatar_option = st.sidebar.radio(
    "Avatar Presence:",
    ('Avatar Visible', 'No Avatar'),
    index=current_avatar_index,
    key='avatar_radio',
    # disabled=sidebar_disabled # Uncomment to lock after start
)
lsm_option = st.sidebar.radio(
    "Linguistic Style:",
    ('Adaptive LSM (Simulated)', 'Static Style'),
    index=current_lsm_index,
    key='lsm_radio',
    # disabled=sidebar_disabled # Uncomment to lock after start
)
debug_mode = st.sidebar.checkbox("🔍 Show Debug Info (Style/LSM)", value=False, key='debug_check')

# Update session state if changes are made *before* disclaimer is accepted
# (Prevents changing conditions mid-conversation easily)
if not sidebar_disabled:
    show_avatar_setting = (avatar_option == 'Avatar Visible')
    use_adaptive_lsm = (lsm_option == 'Adaptive LSM (Simulated)')

    condition_changed = False
    if st.session_state.experiment_condition.get('avatar') != show_avatar_setting:
        st.session_state.experiment_condition['avatar'] = show_avatar_setting
        condition_changed = True
    if st.session_state.experiment_condition.get('lsm') != use_adaptive_lsm:
        st.session_state.experiment_condition['lsm'] = use_adaptive_lsm
        condition_changed = True

    # # Optional: If condition changes before start, maybe reset disclaimer?
    # if condition_changed:
    #      st.session_state.disclaimer_step = 1
    #      # No rerun here, let the natural flow handle it

st.sidebar.markdown("---")
st.sidebar.write(f"**Participant ID:** {st.session_state.participant_id if st.session_state.participant_id else 'Not Set'}")
st.sidebar.write(f"**Assigned Condition:**")
st.sidebar.write(f"- Avatar: {'Yes' if st.session_state.experiment_condition.get('avatar') else 'No'}")
st.sidebar.write(f"- Style: {'Adaptive' if st.session_state.experiment_condition.get('lsm') else 'Static'}")
st.sidebar.write(f"**Log File:** {os.path.basename(st.session_state.log_file_path) if st.session_state.log_file_path else 'Not Set'}")


# --- Main Application Flow ---

# 1. Check for Participant ID
if not st.session_state.participant_id_entered:
    display_participant_id_input()

# 2. Check for Disclaimer Acceptance (only if ID entered)
elif not st.session_state.disclaimer_accepted:
    display_welcome_disclaimer(st.session_state.experiment_condition['avatar'])

# 3. Run the Chat Application (only if ID entered and disclaimer accepted)
else:
    st.title(f"Chat with {DEFAULT_BOT_NAME if st.session_state.experiment_condition['avatar'] else NO_AVATAR_BOT_NAME}")

    # --- Timeout Check ---
    elapsed_time = time.time() - st.session_state.start_time
    remaining_time = SESSION_TIMEOUT_SECONDS - elapsed_time

    if remaining_time <= 0 and not st.session_state.session_timed_out:
        st.session_state.session_timed_out = True
        st.warning("The session time limit has been reached. Thank you for your participation!")
        if LOG_SESSION_EVENTS:
            log_event({
                "event_type": "session_timeout",
                "duration_seconds": elapsed_time
            })
        # Consider adding a final message or survey link here
        st.stop() # Stop further execution below this point for this run

    # Display remaining time
    st.caption(f"Time remaining: {int(remaining_time // 60)} min {int(remaining_time % 60)} sec")
    if st.session_state.session_timed_out:
        st.info("Session ended due to time limit. Input is disabled.")


    # --- Display Chat History ---
    for i, msg in enumerate(st.session_state.messages):
        avatar = msg.get("small_avatar_used") # Use stored avatar path or None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            # Optionally show debug info stored with user messages
            if debug_mode and msg["role"] == "user" and "style_profile" in msg:
                 with st.expander("Detected User Style Traits (Turn " + str(msg.get('turn_number', '?')) + ")", expanded=False):
                      st.json(msg["style_profile"])
            # Optionally show LSM score calculated *after* the bot response this message follows
            if debug_mode and msg["role"] == "assistant" and "lsm_score" in msg:
                 st.caption(f"LSM vs Prev User Msg: {msg['lsm_score']:.3f}")


    # --- Handle User Input ---
    # Disable input if session has timed out
    chat_input_disabled = st.session_state.session_timed_out
    prompt = st.chat_input("What health questions do you have?", disabled=chat_input_disabled, key="chat_input")

    if prompt:
        current_turn = len([m for m in st.session_state.messages if m['role'] == 'user']) + 1

        # 1. Detect User Style for CURRENT prompt
        profile = detect_style_traits(prompt)

        # 2. Prepare for API call: Add PREVIOUS smoothed LSM score to the profile
        # This is the score that influences the *upcoming* bot response style
        if st.session_state.smoothed_lsm_score is not None:
             profile["lsm_score_prev"] = st.session_state.smoothed_lsm_score
        else:
             # Default for the very first turn if needed
             profile["lsm_score_prev"] = 0.5

        # 3. Display User Message & Log (with profile containing PREVIOUS smoothed LSM)
        avatar_to_display = None # Users don't have an avatar in this setup
        with st.chat_message("user", avatar=avatar_to_display):
            st.markdown(prompt)
            if debug_mode:
                # Display the detected traits, including the smoothed LSM score used for the *next* bot response
                with st.expander(f"Detected User Style Traits (Turn {current_turn}) & Prev Smoothed LSM", expanded=True):
                    st.json(profile)

        # Add user message to state *before* calling the API
        # Profile stored here includes the 'lsm_score_prev' used for the next bot generation
        user_message_data = {
            "role": "user",
            "content": prompt,
            "style_profile": profile,
            "turn_number": current_turn
        }
        st.session_state.messages.append(user_message_data)

        # Log user message event
        if LOG_SESSION_EVENTS:
             log_event({
                "event_type": "user_message",
                "turn_number": current_turn,
                "user_prompt": prompt,
                "style_profile": profile, # Contains prev smoothed LSM
                "condition": st.session_state.experiment_condition
             })

        # 4. Generate Bot Response
        with st.chat_message("assistant", avatar=(AVATAR_IMAGE_PATH if st.session_state.experiment_condition['avatar'] else None)):
            with st.spinner("HopDoc is thinking..."):
                # Get response and the system instruction actually used
                bot_response, system_instr_used = get_gemini_response(
                    user_prompt=prompt,
                    # History includes all messages *before* the current user prompt was appended
                    chat_history=st.session_state.messages[:-1],
                    is_adaptive=st.session_state.experiment_condition["lsm"],
                    show_avatar=st.session_state.experiment_condition["avatar"],
                    # Pass the profile containing the previous smoothed LSM score
                    style_profile=profile
                )

                # --- Calculate and Smooth LSM FOR THIS TURN ---
                # Tokenize texts for length checks and score calculation
                user_tokens = tokenize_text(prompt)
                bot_tokens = tokenize_text(bot_response)
                user_token_count = len(user_tokens)
                bot_token_count = len(bot_tokens)

                # Calculate the raw LSM score for THIS turn (user vs bot)
                raw_turn_lsm = compute_lsm_score(prompt, bot_response)

                # Store raw score in session state temporarily for logging/display with bot message
                st.session_state.lsm_score = raw_turn_lsm

                # Determine if smoothed score should be updated
                update_smoothed_lsm = True
                lsm_reason = ""

                if user_token_count < MIN_LSM_TOKENS:
                    update_smoothed_lsm = False
                    lsm_reason += f"User ({user_token_count}) < {MIN_LSM_TOKENS} tokens. "
                if bot_token_count < MIN_LSM_TOKENS:
                    update_smoothed_lsm = False
                    lsm_reason += f"Bot ({bot_token_count}) < {MIN_LSM_TOKENS} tokens."

                # Calculate the new smoothed LSM score (used for the *next* prompt)
                if update_smoothed_lsm:
                    prev_smoothed_lsm = st.session_state.smoothed_lsm_score
                    new_smoothed_lsm = (LSM_SMOOTHING_ALPHA * raw_turn_lsm) + ((1 - LSM_SMOOTHING_ALPHA) * prev_smoothed_lsm)
                    st.session_state.smoothed_lsm_score = new_smoothed_lsm # Update state
                    lsm_reason = f"Smoothed Score Updated: {new_smoothed_lsm:.3f}"
                else:
                    lsm_reason += "Smoothed Score Not Updated."
                    # Keep the previous value if not updated
                    new_smoothed_lsm = st.session_state.smoothed_lsm_score

            # Display bot response
            st.markdown(bot_response)
            if debug_mode:
                st.caption(f"Raw Turn LSM: {raw_turn_lsm:.3f} | {lsm_reason}")
                with st.expander("System Prompt Used", expanded=False):
                     st.text(system_instr_used)

        # 5. Store Bot Message & Log
        bot_message_data = {
             "role": "assistant",
             "content": bot_response,
             "lsm_score": raw_turn_lsm, # Log the RAW score calculated for this turn
             "smoothed_lsm_after_turn": new_smoothed_lsm, # Log the smoothed value AFTER this turn
             "system_instruction_used": system_instr_used,
             "turn_number": current_turn,
             "small_avatar_used": AVATAR_IMAGE_PATH if st.session_state.experiment_condition['avatar'] else None
        }
        st.session_state.messages.append(bot_message_data)

        if LOG_SESSION_EVENTS:
             log_event({
                 "event_type": "bot_response",
                 "turn_number": current_turn,
                 "user_prompt": prompt,
                 "style_profile": profile, # Profile used for THIS response generation
                 "lsm_score_raw_turn": raw_turn_lsm,
                 "smoothed_lsm_after_turn": new_smoothed_lsm, # Log the state *after* this turn
                 "lsm_update_reason": lsm_reason,
                 "system_instruction_used": system_instr_used,
                 "bot_response_processed": bot_response,
                 "condition": st.session_state.experiment_condition
             })
        st.rerun()