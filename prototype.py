"""
Streamlit web application for my MS Thesis Experiment.

This app simulates a health chatbot (HopDoc) with different conditions
to test user perceptions based on avatar presence and linguistic style matching (LSM).
It uses the Google Gemini API for generating responses and includes an enhanced
LSM simulation based on detected user style traits.
"""

import streamlit as st
import time
import random
import google.generativeai as genai
import os
import re # Needed for regex-based style detection

# --- Page Configuration (Must be the first Streamlit command!) ---
st.set_page_config(layout="wide")

# --- Global Configuration & Constants ---
AVATAR_IMAGE_PATH = "franko.png"
DEFAULT_BOT_NAME = "HopDoc (Franko)"
NO_AVATAR_BOT_NAME = "HopDoc"
# Using the specific experimental model I selected.
GEMINI_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"
SESSION_TIMEOUT_SECONDS = 600 # 10 minutes for the interaction

# --- Lists for Style Detection ---
INFORMAL_WORDS = r"\b(bruh|yo|dude|lol|lmao|deadass|ain't|gonna|wanna|gotta|kinda|sorta|lemme|dunno|ya|nah|tho|fr|btw|imo|idk|omg)\b"
HEDGING_WORDS = r"\b(maybe|not sure|kinda|sort of|might|possibly|perhaps|seems|appears|suggests|i think|i guess|idk)\b"
EMOJI_PATTERN = r"[😀-🙏💪😎😉😂🤣😭😍❤️✨⭐👍👎🤔🥳]" # Basic emoji range, can be expanded
POSITIVE_WORDS = {"good", "great", "awesome", "thanks", "helpful", "like", "love", "nice", "cool", "perfect", "excellent", "amazing"}
NEGATIVE_WORDS = {"bad", "terrible", "hate", "problem", "issue", "difficult", "not helpful", "sucks", "annoying", "frustrating"}

# --- API Key & Gemini Client Setup ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    st.error("🚨 Oops! Your `GOOGLE_API_KEY` was not found in the Streamlit secrets. Please add it.")
    st.stop()
except Exception as e:
    st.error(f"🚨 An error occurred while configuring the Gemini API: {e}")
    st.stop()

# --- NEW: Style Detection & Prompt Generation Functions ---

def detect_style_traits(user_input):
    """
    Analyzes user input using regex/simple heuristics and returns a dict of detected style traits.
    This helps ground the LSM simulation.
    """
    lower_input = user_input.lower()
    words = set(re.findall(r'\b\w+\b', lower_input)) # Get unique words

    traits = {
        "emoji": bool(re.search(EMOJI_PATTERN, user_input)),
        "informal": bool(re.search(INFORMAL_WORDS, lower_input, re.IGNORECASE)),
        "hedging": bool(re.search(HEDGING_WORDS, lower_input, re.IGNORECASE)),
        "questioning": user_input.strip().endswith("?"),
        "exclamatory": user_input.count("!") > 0,
        "short": len(user_input.split()) <= 10, # Simple word count threshold
        "positive_sentiment": bool(words.intersection(POSITIVE_WORDS)),
        "negative_sentiment": bool(words.intersection(NEGATIVE_WORDS)),
        "pronouns": {
            "i": bool(re.search(r'\bi\b', lower_input)), # Use regex word boundary
            "you": bool(re.search(r'\byou\b', lower_input)),
            "we": bool(re.search(r'\bwe\b', lower_input))
        }
    }
    # Avoid conflicting sentiment flags if both trigger (simple override)
    if traits["positive_sentiment"] and traits["negative_sentiment"]:
         traits["negative_sentiment"] = False # Let positive override for simplicity

    return traits

def generate_dynamic_prompt(base_prompt, style_profile):
    """
    Modifies the base system instruction with specific style adaptation cues
    based on the detected user style traits.
    """
    style_cues = [] # Start with an empty list for adaptation instructions

    # Generate cues based on detected traits
    if style_profile.get("emoji"):
        style_cues.append("Feel free to include relevant emojis (like 🤔, 👍, 🌱) occasionally to match the user's expressive style, but don't overdo it.")
    if style_profile.get("informal"):
        style_cues.append("Adopt a more informal, relaxed tone. You can use contractions (e.g., 'it's', 'don't') and common, friendly slang if appropriate (e.g., 'cool', 'awesome').")
    if style_profile.get("hedging"):
        style_cues.append("Mirror the user's uncertainty by using tentative language (e.g., 'might', 'could', 'maybe', 'perhaps it could be useful to...').")
    if style_profile.get("short"):
        style_cues.append("Keep your sentences relatively brief and to the point, similar to the user.")
    if style_profile.get("questioning"):
        style_cues.append("Since the user asked a question, ensure your response directly addresses it. You can ask a clarifying follow-up question if needed.")
    if style_profile.get("exclamatory"):
        style_cues.append("You can use an exclamation point for enthusiasm if it fits the context (e.g., 'That's a great idea!'), but use them sparingly.")
    if style_profile.get("positive_sentiment"):
        style_cues.append("Reflect the user's positive tone with encouraging and optimistic language.")
    if style_profile.get("negative_sentiment"):
        style_cues.append("Acknowledge any user frustration with empathy (e.g., 'I understand that can be frustrating...'), then gently guide towards a constructive suggestion.")

    # Pronoun cueing (simple examples)
    pronouns = style_profile.get("pronouns", {})
    if pronouns.get("we"):
        style_cues.append("Consider using inclusive language like 'we' or 'let's' when suggesting actions (e.g., 'Maybe we could explore...').")
    elif pronouns.get("you"):
        style_cues.append("Frame suggestions directly using 'you' (e.g., 'You could try...').")
    elif pronouns.get("i"):
        # Be cautious with 'I' for the bot unless expressing empathy
        style_cues.append("Use empathetic phrasing like 'I understand...' or 'I hear you...' if appropriate.")

    # Combine base prompt with cues if any were generated
    if style_cues:
        # Add a clear separator and list the cues
        modified_prompt = base_prompt + "\n\nIMPORTANT Style Adaptation Instructions (apply subtly):\n- " + "\n- ".join(style_cues)
    else:
        # No specific cues detected, use the base adaptive prompt
        modified_prompt = base_prompt

    return modified_prompt

# --- Core Functions ---

# MODIFIED function to accept style_profile and use dynamic prompt generation
def get_gemini_response(user_prompt, chat_history, is_adaptive, show_avatar, style_profile=None):
    """
    Fetches a response from the configured Gemini model.

    Uses dynamic system prompts for the 'Adaptive LSM' condition based on
    the detected style profile of the user's last message.
    """
    persona_name = DEFAULT_BOT_NAME if show_avatar else NO_AVATAR_BOT_NAME

    # --- Define BASE System Prompts ---
    # Base prompt for the adaptive condition (cues will be added dynamically)
    base_adaptive_prompt = f"""
    You are {persona_name}, a friendly AI wellness assistant. Your goal is to provide helpful, general health and wellness information (fitness, nutrition, sleep, stress management).
    Your tone should feel approachable, supportive, and informative.
    You need to adapt your response style based on the user's last message, following the specific instructions below.
    Keep responses relevant to health and wellness. Do NOT provide medical advice or diagnoses.
    Remember the disclaimer: you are not a medical professional. If asked for medical advice, politely decline and state your limitations.
    """
    # Fixed prompt for the static condition
    static_prompt = f"""
    You are {persona_name}, a friendly AI wellness assistant. Your goal is to provide helpful, general health and wellness information (fitness, nutrition, sleep, stress management).
    Your communication style is CONSISTENTLY: Empathetic, clear, encouraging, and informative, using moderate sentence length (around 1-3 sentences per response). Use standard, proper language without slang or excessive informality. Avoid emojis.
    Maintain this specific style REGARDLESS of the user's writing style. Do NOT try to mimic the user.
    Keep responses relevant to health and wellness. Do NOT provide medical advice or diagnoses.
    Remember the disclaimer: you are not a medical professional. If asked for medical advice, politely decline and state your limitations.
    """

    # --- Determine Final System Instruction ---
    if is_adaptive and style_profile:
        # Generate the full prompt with dynamic style cues
        system_instruction_text = generate_dynamic_prompt(base_adaptive_prompt, style_profile)
    else:
        # Use the fixed static prompt if not adaptive or no profile provided
        system_instruction_text = static_prompt

    # --- Format Chat History for API ---
    messages_for_api = []
    for message in chat_history:
        role = "model" if message["role"] == "assistant" else message["role"]
        messages_for_api.append({"role": role, "parts": [{"text": message["content"]}]})
    messages_for_api.append({"role": "user", "parts": [{"text": user_prompt}]}) # Add latest user prompt

    # --- Call the Gemini API ---
    try:
        model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            system_instruction=system_instruction_text # Use the determined prompt
        )
        response = model.generate_content(messages_for_api)

        if not response.candidates or not response.candidates[0].content.parts:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 return f"Response blocked due to: {response.prompt_feedback.block_reason}. Let's try a different topic."
             else:
                 return "Sorry, I couldn't generate a response for that. Could you rephrase or ask about something else?"

        return response.text

    except Exception as e:
        print(f"Gemini API Error Details: {type(e).__name__} - {e}")
        return "Sorry, I encountered an error trying to respond. Please try again."


# --- Welcome/Disclaimer Function --- (No changes needed inside, calls are updated later)
def display_welcome_disclaimer(show_avatar_in_welcome):
    """ Handles the multi-step welcome message and disclaimer presentation. """
    if 'disclaimer_step' not in st.session_state:
        st.session_state.disclaimer_step = 1

    col1, col2 = st.columns([0.8, 0.2], gap="medium")

    with col1:
        st.title("Welcome to HopDoc")
        if st.session_state.disclaimer_step == 1:
            st.markdown(f"""Hi there! I'm **Franko** – your friendly, fitness-focused frog! {'*(Avatar simulated)*' if not show_avatar_in_welcome else ''}\n\nOver the next **{int(SESSION_TIMEOUT_SECONDS / 60)} minutes**, feel free to ask me any questions related to health and wellness – like fitness tips, nutrition, sleep, or managing stress. Try to keep our conversation on-topic and appropriate.""")
            if st.button("Continue"):
                st.session_state.disclaimer_step = 2
                st.rerun()
        elif st.session_state.disclaimer_step == 2:
            st.markdown("""**⚠️ Just a heads up:**\n\nI'm **not** a medical professional, and I **can't** provide medical advice, diagnoses, or emergency support. If you're experiencing a health crisis or need medical care, **please contact a healthcare provider or emergency services.**""")
            if st.button("Continue "):
                st.session_state.disclaimer_step = 3
                st.rerun()
        elif st.session_state.disclaimer_step == 3:
            st.markdown("""Not sure where to start? You could try asking:\n\n*   🤔 *"How can I improve my sleep routine?"*\n*   🧘 *"What are quick ways to reduce stress during the day?"*\n\nHave fun, and let's leap into wellness together!""")
            if st.button("Let's Begin!"):
                st.session_state.disclaimer_accepted = True
                st.session_state.start_time = time.time()
                st.session_state.messages = [] # Clear chat history

                # --- Generate initial greeting ---
                # For the initial greeting, we don't have user input yet, so no style profile.
                # We'll use the base prompt appropriate for the starting condition.
                initial_greeting = get_gemini_response(
                    user_prompt="<User just started the chat>", # Placeholder
                    chat_history=[],
                    is_adaptive=st.session_state.experiment_condition['lsm'],
                    show_avatar=st.session_state.experiment_condition['avatar'],
                    style_profile=None # No profile for initial greeting
                )
                small_avatar_icon = AVATAR_IMAGE_PATH if st.session_state.experiment_condition['avatar'] else None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": initial_greeting,
                    "small_avatar_used": small_avatar_icon
                    # No style profile for the bot's initial message
                })
                st.rerun()

    with col2:
        if show_avatar_in_welcome and st.session_state.disclaimer_step <= 3:
             try:
                st.image(AVATAR_IMAGE_PATH, width=100)
             except FileNotFoundError:
                 st.warning(f"Avatar image '{AVATAR_IMAGE_PATH}' not found.")
             except Exception as e:
                 st.error(f"Error loading welcome avatar: {e}")

# --- Initialize Session State ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'disclaimer_accepted' not in st.session_state: st.session_state.disclaimer_accepted = False
if 'experiment_condition' not in st.session_state: st.session_state.experiment_condition = {'avatar': True, 'lsm': True}
if 'start_time' not in st.session_state: st.session_state.start_time = None

# --- Sidebar: Experiment Controls ---
st.sidebar.title("Experiment Setup")
st.sidebar.write("*(Control Panel for Researcher)*")
avatar_option = st.sidebar.radio("Avatar Presence:", ('Avatar Visible', 'No Avatar'), index=0 if st.session_state.experiment_condition['avatar'] else 1, key="avatar_radio")
lsm_option = st.sidebar.radio("Linguistic Style:", ('Adaptive LSM (Simulated)', 'Static Style'), index=0 if st.session_state.experiment_condition['lsm'] else 1, key="lsm_radio")

# --- NEW: Debug Mode Toggle ---
debug_mode = st.sidebar.checkbox("🔍 Show Detected Style Traits (Pilot Mode)", value=False, key="debug_mode")

show_avatar_setting = (avatar_option == 'Avatar Visible')
use_adaptive_lsm_setting = (lsm_option == 'Adaptive LSM (Simulated)')

if st.session_state.experiment_condition['avatar'] != show_avatar_setting or \
   st.session_state.experiment_condition['lsm'] != use_adaptive_lsm_setting:
    st.session_state.experiment_condition['avatar'] = show_avatar_setting
    st.session_state.experiment_condition['lsm'] = use_adaptive_lsm_setting
    # Consider resetting chat/timer if conditions change mid-interaction during testing

st.sidebar.markdown("---")
st.sidebar.write(f"**Current Condition:**")
st.sidebar.write(f"- Avatar: {'Yes' if st.session_state.experiment_condition['avatar'] else 'No'}")
st.sidebar.write(f"- Style: {'Adaptive (Simulated)' if st.session_state.experiment_condition['lsm'] else 'Static'}")

# --- Main App Interface ---
if not st.session_state.disclaimer_accepted:
    display_welcome_disclaimer(show_avatar_in_welcome=st.session_state.experiment_condition['avatar'])
else:
    st.title("Chat with HopDoc")
    time_limit_reached = False
    if st.session_state.start_time:
        elapsed_time = time.time() - st.session_state.start_time
        minutes, seconds = divmod(elapsed_time, 60)
        timer_display = f"Time elapsed: {int(minutes):02}:{int(seconds):02} / {int(SESSION_TIMEOUT_SECONDS / 60):02}:00"
        time_limit_reached = elapsed_time > SESSION_TIMEOUT_SECONDS
    else:
        timer_display = "Timer not started."

    avatar_col, chat_col = st.columns([0.3, 0.7], gap="large")

    with avatar_col: # Avatar column
        if st.session_state.experiment_condition['avatar']:
            try: st.image(AVATAR_IMAGE_PATH, use_container_width=True)
            except FileNotFoundError: st.warning(f"Large avatar image '{AVATAR_IMAGE_PATH}' not found.")
            except Exception as e: st.error(f"Error loading large avatar: {e}")
        else: st.write("")

    with chat_col: # Chat column
        st.caption(timer_display)
        if time_limit_reached:
            st.warning("Interaction time limit reached. Please complete the survey.")

        use_small_avatar_icon = None # Not using small icon if large avatar present

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("small_avatar_used")):
                st.markdown(message["content"])
                # --- NEW: Optionally display stored style profile in debug mode ---
                if debug_mode and message["role"] == "user" and "style_profile" in message:
                    st.caption("Detected Style:")
                    st.json(message["style_profile"], expanded=False)


        # Handle user input
        if prompt := st.chat_input("What health questions do you have?", disabled=time_limit_reached, key="chat_input"):

            # --- Detect user style FIRST ---
            user_style_profile = detect_style_traits(prompt)

            # 1. Display and store user message (including style profile)
            with st.chat_message("user"):
                st.markdown(prompt)
                # Display detected traits under user message if debug mode is on
                if debug_mode:
                    st.caption("Detected Style:")
                    st.json(user_style_profile, expanded=False)

            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "small_avatar_used": None,
                "style_profile": user_style_profile # Store detected profile
            })

            # --- Display detected traits in sidebar if debug mode is on ---
            if debug_mode:
                 st.sidebar.markdown("---")
                 st.sidebar.markdown("### 🧠 Last User Style Detected")
                 st.sidebar.json(user_style_profile)


            # 2. Generate and display assistant response
            with st.chat_message("assistant", avatar=use_small_avatar_icon):
                with st.spinner("Thinking..."):
                    # --- Pass detected style profile to the LLM function ---
                    response = get_gemini_response(
                        prompt, # Pass current prompt for context if needed by function logic
                        chat_history=st.session_state.messages[:-1], # Pass history *before* this user turn
                        is_adaptive=st.session_state.experiment_condition['lsm'],
                        show_avatar=st.session_state.experiment_condition['avatar'],
                        style_profile=user_style_profile # Pass the detected profile
                    )
                    st.markdown(response)

            # 3. Store assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "small_avatar_used": use_small_avatar_icon
                # Could add bot style profile here later if needed
            })

            st.rerun() # Rerun for immediate UI update