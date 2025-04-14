"""
Streamlit web application for my MS Thesis Experiment.

This app simulates a health chatbot (HopDoc) with different conditions
to test user perceptions based on avatar presence and linguistic style matching (LSM).
It uses the Google Gemini API for generating responses.
"""

import streamlit as st
import time
import random
import google.generativeai as genai
import os # Needed for path joining if debugging secrets, but primarily using st.secrets

# --- Page Configuration (Must be the first Streamlit command!) ---
# I'm setting the layout to "wide" to give the elements more space.
st.set_page_config(layout="wide")

# --- Global Configuration & Constants ---
# Here I define the core settings and filenames for the app.
AVATAR_IMAGE_PATH = "franko.png" # Path to the main Franko avatar image
DEFAULT_BOT_NAME = "HopDoc (Franko)" # Name used when avatar is visible
NO_AVATAR_BOT_NAME = "HopDoc" # Name used when no avatar is shown
# Using the specific experimental model I selected.
GEMINI_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"
SESSION_TIMEOUT_SECONDS = 600 # 10 minutes for the interaction

# --- API Key & Gemini Client Setup ---
# I need to configure the Gemini client using the API key.
# I'm using Streamlit's secrets management for security.
try:
    # This tries to load the key from `.streamlit/secrets.toml`
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    # You could add a silent success log here if needed for deployment checks
    # print("Gemini API Key configured successfully.")
except KeyError:
    # This error means the secrets file exists, but the key isn't IN it.
    st.error("🚨 Oops! Your `GOOGLE_API_KEY` was not found in the Streamlit secrets. Please add it.")
    st.stop() # Stop the app if the key is missing.
except Exception as e:
    # Catch any other unexpected errors during configuration.
    st.error(f"🚨 An error occurred while configuring the Gemini API: {e}")
    st.stop()

# --- Core Functions ---

def get_gemini_response(user_prompt, chat_history, is_adaptive, show_avatar):
    """
    Fetches a response from the configured Gemini model.

    I engineered the system prompts here to simulate the different experimental
    conditions (Static vs. Adaptive Linguistic Style).
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

    # Format the chat history for the Gemini API.
    # It expects a specific structure with 'role' and 'parts'.
    messages_for_api = []
    for message in chat_history:
        role = "model" if message["role"] == "assistant" else message["role"]
        messages_for_api.append({"role": role, "parts": [{"text": message["content"]}]})

    # Add the latest user prompt in the required format
    messages_for_api.append({"role": "user", "parts": [{"text": user_prompt}]})

    # Call the Gemini API
    try:
        # Initialize the model with the appropriate system instruction
        model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            system_instruction=system_instruction_text
        )
        # Generate the response based on the formatted history + new prompt
        response = model.generate_content(messages_for_api)

        # I need to handle cases where the API might block the response or return nothing.
        if not response.candidates or not response.candidates[0].content.parts:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 # Inform the user if the response was blocked for safety reasons
                 return f"Response blocked due to: {response.prompt_feedback.block_reason}. Let's try a different topic."
             else:
                 # Generic fallback if no response content is found
                 return "Sorry, I couldn't generate a response for that. Could you rephrase or ask about something else?"

        # If successful, return the generated text content
        return response.text

    except Exception as e:
        # Log the detailed error to the console during development/debugging
        print(f"Gemini API Error Details: {type(e).__name__} - {e}")
        # Provide a user-friendly error message in the chat interface
        return "Sorry, I encountered an error trying to respond. Please try again."


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

                # Generate the initial greeting from the bot based on the starting condition
                initial_greeting = get_gemini_response(
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
    # Stores the chat history (list of dicts)
    st.session_state.messages = []
if 'disclaimer_accepted' not in st.session_state:
    # Tracks if the user has passed the initial disclaimer screens
    st.session_state.disclaimer_accepted = False
if 'experiment_condition' not in st.session_state:
    # Stores the current experimental setup (avatar T/F, LSM T/F)
    # I'll start with Avatar + Adaptive LSM as the default view.
    st.session_state.experiment_condition = {'avatar': True, 'lsm': True}
if 'start_time' not in st.session_state:
    # Records when the user clicks "Let's Begin!" to track interaction time
    st.session_state.start_time = None

# --- Sidebar: Experiment Controls (For Researcher) ---
# This section allows me (the researcher) to easily switch conditions.
st.sidebar.title("Experiment Setup")
st.sidebar.write("*(Control Panel for Researcher)*")

# Radio buttons to select the Avatar Presence condition
avatar_option = st.sidebar.radio(
    "Avatar Presence:",
    ('Avatar Visible', 'No Avatar'),
    # Set default based on session state
    index=0 if st.session_state.experiment_condition['avatar'] else 1,
    key="avatar_radio" # Added key for stability
)
# Radio buttons to select the Linguistic Style condition
lsm_option = st.sidebar.radio(
    "Linguistic Style:",
    ('Adaptive LSM (Simulated)', 'Static Style'),
    # Set default based on session state
    index=0 if st.session_state.experiment_condition['lsm'] else 1,
    key="lsm_radio" # Added key for stability
)

# Determine current settings based on radio button selections
show_avatar_setting = (avatar_option == 'Avatar Visible')
use_adaptive_lsm_setting = (lsm_option == 'Adaptive LSM (Simulated)')

# Update the session state ONLY if a condition has actually changed
# This prevents unnecessary resets or reruns if the user just re-clicks the same option.
if st.session_state.experiment_condition['avatar'] != show_avatar_setting or \
   st.session_state.experiment_condition['lsm'] != use_adaptive_lsm_setting:
    st.session_state.experiment_condition['avatar'] = show_avatar_setting
    st.session_state.experiment_condition['lsm'] = use_adaptive_lsm_setting
    # NOTE: Changing condition mid-chat might be confusing for the user/data.
    # For a real experiment, I'd likely assign conditions *before* the user starts
    # and potentially disable these controls during the participant interaction.
    # Consider uncommenting below if a reset is desired on condition change:
    # st.session_state.messages = []
    # st.session_state.start_time = None # Reset timer too?
    # st.rerun()

# Display the currently active condition for confirmation
st.sidebar.markdown("---")
st.sidebar.write(f"**Current Condition:**")
st.sidebar.write(f"- Avatar: {'Yes' if st.session_state.experiment_condition['avatar'] else 'No'}")
st.sidebar.write(f"- Style: {'Adaptive (Simulated)' if st.session_state.experiment_condition['lsm'] else 'Static'}")


# --- Main App Interface ---

# Show disclaimer first, then the chat interface
if not st.session_state.disclaimer_accepted:
    # Pass the current avatar setting to the disclaimer display function
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
        # Should ideally not happen if disclaimer_accepted is True, but safer
        timer_display = "Timer not started."

    # Define the main layout with columns for the avatar and the chat
    avatar_col, chat_col = st.columns([0.3, 0.7], gap="large") # Adjust ratio as needed

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
            # Keep the column structure even if empty
            st.write("")

    # --- Chat Column ---
    with chat_col:
        # Display the timer within the chat column
        st.caption(timer_display)
        if time_limit_reached:
            st.warning("Interaction time limit reached. Please complete the survey.")

        # Determine the small icon to use next to assistant messages
        # I decided *not* to show the small icon if the large avatar is present,
        # but uncomment the first line if you prefer to show both.
        # use_small_avatar_icon = AVATAR_IMAGE_PATH if st.session_state.experiment_condition['avatar'] else None
        use_small_avatar_icon = None # Set to None if large avatar is primary visual

        # Display existing chat messages from history
        for message in st.session_state.messages:
            # Use the 'small_avatar_used' key stored with the message for consistency
            with st.chat_message(message["role"], avatar=message.get("small_avatar_used")):
                st.markdown(message["content"])

        # Get user input using the chat input widget
        # Disable input if the time limit has been reached
        if prompt := st.chat_input("What health questions do you have?", disabled=time_limit_reached, key="chat_input"):

            # 1. Display and store the user's message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "small_avatar_used": None # User messages don't have an avatar icon
                })

            # 2. Generate and display the assistant's response
            with st.chat_message("assistant", avatar=use_small_avatar_icon):
                # Show a thinking indicator while waiting for the API
                with st.spinner("Thinking..."):
                    response = get_gemini_response(
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
                "small_avatar_used": use_small_avatar_icon # Store the icon used
                })

            # Rerun the script immediately after processing input/output.
            # This ensures the new messages are displayed correctly and the input box is ready.
            st.rerun()