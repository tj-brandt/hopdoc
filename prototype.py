"""
Streamlit web application for my MS Thesis Experiment.

This app simulates a health chatbot (HopDoc) with different conditions
to test user perceptions based on avatar presence and linguistic style matching (LSM).
It uses OpenRouter's API to access the specified LLM for generating responses.
"""

import streamlit as st
import time
import random
import requests # Using requests library for OpenRouter API calls
import os

# --- Page Configuration (Must be the first Streamlit command!) ---
st.set_page_config(layout="wide")

# --- Global Configuration & Constants ---
# Here I define the core settings and filenames for the app.
AVATAR_IMAGE_PATH = "franko.png" # Path to the main Franko avatar image
DEFAULT_BOT_NAME = "HopDoc (Franko)" # Name used when avatar is visible
NO_AVATAR_BOT_NAME = "HopDoc" # Name used when no avatar is shown

# --- OpenRouter Configuration ---
OPENROUTER_MODEL_NAME = "google/gemini-2.5-pro-exp-03-25:free" # Your selected OpenRouter model
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Your Streamlit app's URL - REQUIRED for the HTTP-Referer header by OpenRouter
APP_URL = "https://hopdoc.streamlit.app/"
# --- End OpenRouter Config ---

SESSION_TIMEOUT_SECONDS = 600 # 10 minutes for the interaction

# --- API Key Setup (Using OpenRouter Key) ---
# I need the OpenRouter API key from Streamlit secrets.
try:
    # This tries to load the key from `.streamlit/secrets.toml`
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    # This error means the secrets file exists, but the key isn't IN it.
    st.error("🚨 Oops! Your `OPENROUTER_API_KEY` was not found in the Streamlit secrets. Please add it.")
    st.stop() # Stop the app if the key is missing.
except Exception as e:
    # Catch any other unexpected errors during secret access.
    st.error(f"🚨 An error occurred while accessing secrets: {e}")
    st.stop()

# --- Core Functions ---

def get_openrouter_response(user_prompt, chat_history, is_adaptive, show_avatar):
    """
    Fetches a response from the specified model via OpenRouter's API.

    Uses standard HTTP requests and handles OpenRouter's specific requirements
    like headers and JSON format. System prompts are included as the first message.
    """
    # Determine the persona name based on the avatar condition
    persona_name = DEFAULT_BOT_NAME if show_avatar else NO_AVATAR_BOT_NAME

    # Define the core instructions for the LLM based on the LSM condition
    if is_adaptive:
        system_instruction_text = f"""
        You are {persona_name}. Your goal is to provide helpful, general health and wellness information (fitness, nutrition, sleep, stress management).
        IMPORTANT: Pay close attention to the user's latest message. Analyze its style (e.g., formal/informal, concise/detailed, use of emojis, tone).
        SUBTLY adapt your response style to match the user's style in their *last* message. For example, if they are brief and informal, you be brief and informal. If they are detailed and formal, you be more detailed and formal.
        Maintain your core persona ({persona_name}) but let the user's style influence your expression.
        Keep responses relevant to health and wellness. Do NOT provide medical advice or diagnoses.
        Remember the disclaimer: you are not a medical professional.
        """
    else: # Static Style Condition
        system_instruction_text = f"""
        You are {persona_name}. Your goal is to provide helpful, general health and wellness information (fitness, nutrition, sleep, stress management).
        Your communication style is CONSISTENTLY: Empathetic, clear, encouraging, and informative, using moderate sentence length.
        Maintain this specific style REGARDLESS of the user's writing style. Do NOT try to mimic the user.
        Keep responses relevant to health and wellness. Do NOT provide medical advice or diagnoses.
        Remember the disclaimer: you are not a medical professional.
        """

    # Format messages for OpenRouter (OpenAI compatible format)
    # System instruction goes first
    messages_for_api = [{"role": "system", "content": system_instruction_text}]
    for message in chat_history:
        # Use 'assistant' role for OpenRouter/OpenAI standard
        role = "assistant" if message["role"] == "assistant" else message["role"]
        messages_for_api.append({"role": role, "content": message["content"]})

    # Add the latest user prompt
    messages_for_api.append({"role": "user", "content": user_prompt})

    # Define headers for the OpenRouter API call
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_URL, # Required by OpenRouter
        # Optional: Identify your app - helps OpenRouter analytics
        "X-Title": "HopDoc Thesis Prototype",
    }

    # Define the JSON payload for the request
    payload = {
        "model": OPENROUTER_MODEL_NAME,
        "messages": messages_for_api,
        # Add other parameters like temperature, max_tokens if needed
        # "temperature": 0.7,
        # "max_tokens": 500,
    }

    # Make the API call using the 'requests' library
    try:
        # Set a timeout (e.g., 60 seconds) to prevent hanging indefinitely
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)

        response_data = response.json()

        # Extract the response content - check structure carefully
        if "choices" in response_data and len(response_data["choices"]) > 0:
            # Accessing the nested structure typical of OpenAI compatible APIs
            message_content = response_data["choices"][0]["message"]["content"]
            return message_content.strip() # Remove leading/trailing whitespace
        else:
            # Handle cases where the response structure is unexpected
            print(f"Unexpected OpenRouter response format: {response_data}")
            return "Sorry, I received an unexpected response format. Please try again."

    # Catch specific requests errors
    except requests.exceptions.RequestException as e:
        print(f"OpenRouter API Request Error: {type(e).__name__} - {e}")
        # Provide more specific user feedback based on error type
        if isinstance(e, requests.exceptions.Timeout):
            return "Sorry, the request timed out. The service might be busy. Please try again."
        elif isinstance(e, requests.exceptions.ConnectionError):
            return "Sorry, I couldn't connect to the response service. Please check your network."
        elif isinstance(e, requests.exceptions.HTTPError):
            # Try to get more details from the response body if available
            try:
                error_detail = response.json()
                error_msg = error_detail.get('error', {}).get('message', response.text)
                return f"Sorry, the API returned an error ({response.status_code}): {error_msg}"
            except: # If parsing response fails
                return f"Sorry, the API returned an error: {response.status_code} - {response.reason}"
        # Generic fallback for other request errors
        return "Sorry, an error occurred while communicating with the response service."
    # Catch potential errors during JSON parsing or key access
    except (KeyError, IndexError, Exception) as e:
        print(f"Error processing OpenRouter response: {type(e).__name__} - {e}")
        return "Sorry, I encountered an error processing the response. Please try again."


def display_welcome_disclaimer(show_avatar_in_welcome):
    """
    Handles the multi-step welcome message and disclaimer presentation.
    Uses session state to track the user's progress through the steps.
    """
    # Initialize disclaimer step in session state if it doesn't exist
    if 'disclaimer_step' not in st.session_state:
        st.session_state.disclaimer_step = 1

    # Using columns here to potentially show the avatar next to the welcome text
    col1, col2 = st.columns([0.8, 0.2], gap="medium") # Adjust ratio/gap as needed

    with col1:
        st.title("Welcome to HopDoc")

        # Display content based on the current step
        if st.session_state.disclaimer_step == 1:
            st.markdown(f"""
            Hi there! I'm **Franko** – your friendly, fitness-focused frog! {'*(Avatar simulated)*' if not show_avatar_in_welcome else ''}

            Over the next **{int(SESSION_TIMEOUT_SECONDS / 60)} minutes**, feel free to ask me any questions related to health and wellness – like fitness tips, nutrition, sleep, or managing stress. Try to keep our conversation on-topic and appropriate.
            """)
            # Advance to the next step when the button is clicked
            if st.button("Continue"):
                st.session_state.disclaimer_step = 2
                st.rerun() # Rerun immediately to show the next step

        elif st.session_state.disclaimer_step == 2:
            st.markdown("""
            **⚠️ Just a heads up:**

            I'm **not** a medical professional, and I **can't** provide medical advice, diagnoses, or emergency support. If you're experiencing a health crisis or need medical care, **please contact a healthcare provider or emergency services.**
            """)
            # Using a space in the button label to differentiate from the previous "Continue"
            if st.button("Continue "):
                st.session_state.disclaimer_step = 3
                st.rerun()

        elif st.session_state.disclaimer_step == 3:
            st.markdown("""
            Not sure where to start? You could try asking:

            *   🤔 *"How can I improve my sleep routine?"*
            *   🧘 *"What are quick ways to reduce stress during the day?"*

            Have fun, and let's leap into wellness together!
            """)
            # This button marks the end of the disclaimer and starts the chat
            if st.button("Let's Begin!"):
                st.session_state.disclaimer_accepted = True
                st.session_state.start_time = time.time() # Record the start time
                st.session_state.messages = [] # Ensure chat history is clear

                # Generate the initial greeting from the bot using OpenRouter
                initial_greeting = get_openrouter_response( # <-- UPDATED FUNCTION NAME
                    user_prompt="<User just started the chat>", # Placeholder to trigger a greeting
                    chat_history=[],
                    is_adaptive=st.session_state.experiment_condition['lsm'],
                    show_avatar=st.session_state.experiment_condition['avatar']
                )
                # Determine the small avatar icon to use for the initial message
                small_avatar_icon = AVATAR_IMAGE_PATH if st.session_state.experiment_condition['avatar'] else None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": initial_greeting,
                    # Store the avatar used for this message, consistent with main chat loop
                    "small_avatar_used": small_avatar_icon
                })
                st.rerun() # Rerun to start the main chat interface

    # Display the small avatar image in the second column during the welcome steps
    with col2:
        if show_avatar_in_welcome and st.session_state.disclaimer_step <= 3:
             try:
                st.image(AVATAR_IMAGE_PATH, width=100) # Smaller avatar for welcome
             except FileNotFoundError:
                 st.warning(f"Avatar image '{AVATAR_IMAGE_PATH}' not found.")
             except Exception as e:
                 st.error(f"Error loading welcome avatar: {e}")

# --- Initialize Session State ---
# I need to set up default values for variables that persist across user interactions.
if 'messages' not in st.session_state:
    st.session_state.messages = [] # Stores the chat history
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False # Tracks disclaimer progress
if 'experiment_condition' not in st.session_state:
    # Stores the current experimental setup (avatar T/F, LSM T/F)
    st.session_state.experiment_condition = {'avatar': True, 'lsm': True}
if 'start_time' not in st.session_state:
    st.session_state.start_time = None # Tracks interaction start time

# --- Sidebar: Experiment Controls (For Researcher) ---
st.sidebar.title("Experiment Setup")
st.sidebar.write("*(Control Panel for Researcher)*")

# Radio buttons to select the Avatar Presence condition
avatar_option = st.sidebar.radio(
    "Avatar Presence:", ('Avatar Visible', 'No Avatar'),
    index=0 if st.session_state.experiment_condition['avatar'] else 1,
    key="avatar_radio"
)
# Radio buttons to select the Linguistic Style condition
lsm_option = st.sidebar.radio(
    "Linguistic Style:", ('Adaptive LSM (Simulated)', 'Static Style'),
    index=0 if st.session_state.experiment_condition['lsm'] else 1,
    key="lsm_radio"
)

# Determine current settings based on radio button selections
show_avatar_setting = (avatar_option == 'Avatar Visible')
use_adaptive_lsm_setting = (lsm_option == 'Adaptive LSM (Simulated)')

# Update the session state ONLY if a condition has actually changed
if st.session_state.experiment_condition['avatar'] != show_avatar_setting or \
   st.session_state.experiment_condition['lsm'] != use_adaptive_lsm_setting:
    st.session_state.experiment_condition['avatar'] = show_avatar_setting
    st.session_state.experiment_condition['lsm'] = use_adaptive_lsm_setting
    # Add note about real experiment setup - conditions likely fixed beforehand.

# Display the currently active condition for confirmation
st.sidebar.markdown("---")
st.sidebar.write(f"**Current Condition:**")
st.sidebar.write(f"- Avatar: {'Yes' if st.session_state.experiment_condition['avatar'] else 'No'}")
st.sidebar.write(f"- Style: {'Adaptive (Simulated)' if st.session_state.experiment_condition['lsm'] else 'Static'}")


# --- Main App Interface ---

if not st.session_state.disclaimer_accepted:
    # Show disclaimer screens first
    display_welcome_disclaimer(show_avatar_in_welcome=st.session_state.experiment_condition['avatar'])
else:
    # --- Chat Interface ---
    st.title("Chat with HopDoc")

    # Calculate and display the interaction timer
    time_limit_reached = False
    if st.session_state.start_time:
        elapsed_time = time.time() - st.session_state.start_time
        minutes, seconds = divmod(elapsed_time, 60)
        timer_display = f"Time elapsed: {int(minutes):02}:{int(seconds):02} / {int(SESSION_TIMEOUT_SECONDS / 60):02}:00"
        time_limit_reached = elapsed_time > SESSION_TIMEOUT_SECONDS
    else:
        timer_display = "Timer not started."

    # Define the main layout with columns for the avatar and the chat
    avatar_col, chat_col = st.columns([0.3, 0.7], gap="large")

    # --- Avatar Column ---
    with avatar_col:
        # Display the large Franko avatar if the condition is active
        if st.session_state.experiment_condition['avatar']:
            try:
                st.image(AVATAR_IMAGE_PATH, use_container_width=True)
            except FileNotFoundError:
                st.warning(f"Large avatar image '{AVATAR_IMAGE_PATH}' not found.")
            except Exception as e:
                st.error(f"Error loading large avatar: {e}")
        else:
            st.write("") # Keep column structure

    # --- Chat Column ---
    with chat_col:
        # Display the timer within the chat column
        st.caption(timer_display)
        if time_limit_reached:
            st.warning("Interaction time limit reached. Please complete the survey.")

        # Determine the small icon to use next to assistant messages (set to None)
        use_small_avatar_icon = None

        # Display existing chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("small_avatar_used")):
                st.markdown(message["content"])

        # Get user input using the chat input widget
        if prompt := st.chat_input("What health questions do you have?", disabled=time_limit_reached, key="chat_input"):

            # 1. Display and store the user's message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "small_avatar_used": None
                })

            # 2. Generate and display the assistant's response using OpenRouter
            with st.chat_message("assistant", avatar=use_small_avatar_icon):
                with st.spinner("Thinking..."):
                    response = get_openrouter_response( # <-- UPDATED FUNCTION NAME
                        prompt,
                        st.session_state.messages, # Pass full history
                        is_adaptive=st.session_state.experiment_condition['lsm'],
                        show_avatar=st.session_state.experiment_condition['avatar']
                    )
                    st.markdown(response)

            # 3. Store the assistant's response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "small_avatar_used": use_small_avatar_icon
                })

            # Rerun script for immediate display update
            st.rerun()