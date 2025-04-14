import streamlit as st
# --- SET PAGE LAYOUT FIRST ---
st.set_page_config(layout="wide")
# --- Rest of your imports and code ---
import time
import streamlit as st
import time
import random
import google.generativeai as genai
import os # Might be needed if not using secrets, but secrets is preferred

# --- Configuration ---
AVATAR_IMAGE_PATH = "franko.png"
DEFAULT_BOT_NAME = "HopDoc (Franko)"
NO_AVATAR_BOT_NAME = "HopDoc"
GEMINI_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"

# --- API Key Configuration ---
secrets_path = os.path.join(".streamlit", "secrets.toml")
try:
    # Try loading the API key from Streamlit secrets
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except FileNotFoundError:
    st.error("Secrets file not found error caught. Make sure you have a .streamlit/secrets.toml file.")
    st.stop() # Stop execution if key is not found
except KeyError:
    st.error("GOOGLE_API_KEY not found IN secrets.toml. Please add it.")
    st.stop() # Stop execution if key is not found
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()


# --- LLM Interaction Function ---

def get_gemini_response(user_prompt, chat_history, is_adaptive, show_avatar):
    """Gets a response from the Gemini model based on the prompt, history, and condition."""

    # --- Define System Prompts ---
    persona_name = "Franko the friendly frog from HopDoc" if show_avatar else "HopDoc, a helpful health assistant"
    if is_adaptive:
        system_instruction_text = f"""
        You are {persona_name}. Your goal is to provide helpful, general health and wellness information (fitness, nutrition, sleep, stress management).
        IMPORTANT: Pay close attention to the user's latest message. Analyze its style (e.g., formal/informal, concise/detailed, use of emojis, tone).
        SUBTLY adapt your response style to match the user's style in their *last* message. For example, if they are brief and informal, you be brief and informal. If they are detailed and formal, you be more detailed and formal.
        Maintain your core persona ({persona_name}) but let the user's style influence your expression.
        Keep responses relevant to health and wellness. Do NOT provide medical advice or diagnoses.
        Remember the disclaimer: you are not a medical professional.
        """
    else:
        system_instruction_text = f"""
        You are {persona_name}. Your goal is to provide helpful, general health and wellness information (fitness, nutrition, sleep, stress management).
        Your communication style is CONSISTENTLY: Empathetic, clear, encouraging, and informative, using moderate sentence length.
        Maintain this specific style REGARDLESS of the user's writing style. Do NOT try to mimic the user.
        Keep responses relevant to health and wellness. Do NOT provide medical advice or diagnoses.
        Remember the disclaimer: you are not a medical professional.
        """

    # --- Format Chat History for Gemini API ---
    # Use the 'parts' structure expected by the library
    messages_for_api = []
    for message in chat_history:
        role = "model" if message["role"] == "assistant" else message["role"]
        # Ensure content is correctly nested in 'parts'
        messages_for_api.append({"role": role, "parts": [{"text": message["content"]}]})

    # Add the latest user prompt in the correct format
    messages_for_api.append({"role": "user", "parts": [{"text": user_prompt}]})

    # --- Call the Gemini API ---
    try:
        # Pass system instruction during model initialization - cleaner way
        model = genai.GenerativeModel(
            GEMINI_MODEL_NAME,
            system_instruction=system_instruction_text # Pass system prompt here
        )

        # Generate content using the properly formatted message history
        response = model.generate_content(messages_for_api) # Pass the list of dicts

        # Handle potential safety blocks or empty responses (keep this check)
        if not response.candidates or not response.candidates[0].content.parts:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 return f"Response blocked due to: {response.prompt_feedback.block_reason}. Let's try a different topic."
             else:
                 return "Sorry, I couldn't generate a response for that. Could you rephrase or ask about something else?"

        return response.text

    except Exception as e:
        # Keep the detailed error reporting for now if needed, remove later
        error_details = f"Gemini API Error Details: {type(e).__name__} - {e}"
        print(error_details) # Print to console
        # Optional: Temporarily show detailed error in UI
        # st.error(error_details)
        return "Sorry, I encountered an error trying to respond. Please try again."


# --- Helper Function for Welcome/Disclaimer (Keep as is) ---
def display_welcome_disclaimer(show_avatar_in_welcome):
    # ... (Keep the existing welcome/disclaimer function code) ...
    if 'disclaimer_step' not in st.session_state:
        st.session_state.disclaimer_step = 1

    # Use columns to potentially place avatar next to text if needed
    col1, col2 = st.columns([0.8, 0.2]) # Adjust ratio as needed

    with col1:
        st.title("Welcome to HopDoc")

        if st.session_state.disclaimer_step == 1:
            st.markdown(f"""
            Hi there! I'm **Franko** – your friendly, fitness-focused frog! {'*(Avatar simulated)*' if not show_avatar_in_welcome else ''}

            Over the next **10 minutes**, feel free to ask me any questions related to health and wellness – like fitness tips, nutrition, sleep, or managing stress. Try to keep our conversation on-topic and appropriate.
            """)
            if st.button("Continue"):
                st.session_state.disclaimer_step = 2
                st.rerun() # Rerun to show next step immediately

        elif st.session_state.disclaimer_step == 2:
             st.markdown("""
             **⚠️ Just a heads up:**

             I'm **not** a medical professional, and I **can't** provide medical advice, diagnoses, or emergency support. If you're experiencing a health crisis or need medical care, **please contact a healthcare provider or emergency services.**
             """)
             if st.button("Continue "): # Space distinguishes from prev button if needed
                 st.session_state.disclaimer_step = 3
                 st.rerun()

        elif st.session_state.disclaimer_step == 3:
            st.markdown("""
            Not sure where to start? You could try asking:

            *   🤔 *"How can I improve my sleep routine?"*
            *   🧘 *"What are quick ways to reduce stress during the day?"*

            Or, just click the "**Hop to it!**" button in the chat bar to get a few ideas. *(Prototype Note: Button not implemented yet)*

            Have fun, and let's leap into wellness together!
            """)
            if st.button("Let's Begin!"):
                st.session_state.disclaimer_accepted = True
                st.session_state.start_time = time.time() # Start timer
                # Add initial greeting message from bot AFTER disclaimer
                st.session_state.messages = [] # Clear any previous messages
                # Initial bot greeting depends on condition active *at start*
                initial_greeting = get_gemini_response(
                    user_prompt="<User just started the chat>", # A placeholder to trigger greeting
                    chat_history=[],
                    is_adaptive=st.session_state.experiment_condition['lsm'],
                    show_avatar=st.session_state.experiment_condition['avatar']
                )
                bot_name = DEFAULT_BOT_NAME if st.session_state.experiment_condition['avatar'] else NO_AVATAR_BOT_NAME
                avatar_display = AVATAR_IMAGE_PATH if st.session_state.experiment_condition['avatar'] else None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": initial_greeting,
                    "avatar": avatar_display
                })
                st.rerun()

    with col2:
        if show_avatar_in_welcome and st.session_state.disclaimer_step <= 3:
             try:
                # Display avatar during welcome screens if condition is met
                st.image(AVATAR_IMAGE_PATH, width=100) # Adjust width as needed
             except FileNotFoundError:
                 st.warning("Avatar image not found. Please place 'franko.png' in the same directory.")
             except Exception as e:
                 st.error(f"Error loading image: {e}")


# --- Main App Logic (Initialize Session State - Keep as is) ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'disclaimer_accepted' not in st.session_state:
    st.session_state.disclaimer_accepted = False
if 'experiment_condition' not in st.session_state:
    # Default condition or could be set via URL params later
    st.session_state.experiment_condition = {'avatar': True, 'lsm': True}
if 'start_time' not in st.session_state:
    st.session_state.start_time = None


# --- Condition Selection UI (Sidebar - Keep as is) ---
st.sidebar.title("Experiment Setup")
st.sidebar.write("*(Control Panel for Researcher)*")
avatar_option = st.sidebar.radio(
    "Avatar Presence:",
    ('Avatar Visible', 'No Avatar'),
    index=0 if st.session_state.experiment_condition['avatar'] else 1
)
lsm_option = st.sidebar.radio(
    "Linguistic Style:",
    ('Adaptive LSM (Simulated)', 'Static Style'),
    index=0 if st.session_state.experiment_condition['lsm'] else 1
)
show_avatar = (avatar_option == 'Avatar Visible')
use_adaptive_lsm = (lsm_option == 'Adaptive LSM (Simulated)')
# Update session state IF CHANGED (prevents overwriting initial greeting)
if st.session_state.experiment_condition['avatar'] != show_avatar or \
   st.session_state.experiment_condition['lsm'] != use_adaptive_lsm:
    st.session_state.experiment_condition['avatar'] = show_avatar
    st.session_state.experiment_condition['lsm'] = use_adaptive_lsm
    # Optional: Could reset chat if condition changes mid-way, or just let it adapt
    # st.session_state.messages = [] # Uncomment to reset chat on condition change
    # st.rerun() # Force rerun if condition changed

st.sidebar.markdown("---")
st.sidebar.write(f"**Current Condition:**")
st.sidebar.write(f"- Avatar: {'Yes' if st.session_state.experiment_condition['avatar'] else 'No'}")
st.sidebar.write(f"- Style: {'Adaptive (Simulated)' if st.session_state.experiment_condition['lsm'] else 'Static'}")


# --- Main Interface ---
if not st.session_state.disclaimer_accepted:
    # Use columns for welcome screen too if desired, like before
    display_welcome_disclaimer(show_avatar_in_welcome=st.session_state.experiment_condition['avatar'])
else:
    # --- Chat Interface with Larger Avatar ---
    st.title("Chat with HopDoc")

    # Optional: Display timer (place it before columns or in chat column)
    if st.session_state.start_time:
        elapsed_time = time.time() - st.session_state.start_time
        minutes, seconds = divmod(elapsed_time, 60)
        # Decide where to put the timer caption for best layout
        # st.caption(f"Time elapsed: {int(minutes):02}:{int(seconds):02} / ~10:00")
        time_limit_reached = elapsed_time > 600 # 10 minutes
        if time_limit_reached:
            st.warning("Interaction time limit reached. Please complete the survey.")
    else:
        time_limit_reached = False # Timer hasn't started

    # --- Define Layout Columns ---
    # Adjust the width ratio as needed (e.g., [1, 3] means chat area is 3x wider)
    # Decide if avatar is on lefts ([avatar_col, chat_col]) or right ([chat_col, avatar_col])
    avatar_col, chat_col = st.columns([0.3, 0.7], gap="large") # Example: Avatar takes 30% width

    # --- Avatar Column ---
    with avatar_col:
        if st.session_state.experiment_condition['avatar']:
            try:
                # Display the LARGER avatar image here
                # You might need a different image file for the larger version
                st.image("franko.png", use_container_width=True) # Use preferred parameter
                # Add any other elements you want near the avatar (e.g., nameplate)
                # st.write(f"**{DEFAULT_BOT_NAME}**") # Optional name display
            except FileNotFoundError:
                st.warning("Large avatar image not found.")
            except Exception as e:
                st.error(f"Error loading large image: {e}")
        else:
            # Optionally display a placeholder or leave empty if no avatar condition
            st.write("") # Ensures the column structure is maintained

    # --- Chat Column ---
    with chat_col:
        # Put the timer here if preferred
        if st.session_state.start_time:
             st.caption(f"Time elapsed: {int(minutes):02}:{int(seconds):02} / ~10:00")

        # Determine bot display name and SMALL avatar icon for chat messages
        # You might still want the small icon next to the bubble for clarity
        use_small_avatar_icon = AVATAR_IMAGE_PATH if st.session_state.experiment_condition['avatar'] else None
        # Or maybe you DON'T want the small icon if the big one is present:
        # use_small_avatar_icon = None # Uncomment this if you *only* want the large avatar

        # Display chat messages from history within THIS column
        for message in st.session_state.messages:
            # Use the small icon setting determined above for the message bubble
            with st.chat_message(message["role"], avatar=message.get("small_avatar_used")): # Use the avatar stored with the message
                st.markdown(message["content"])

        # React to user input - place the input box within THIS column
        if prompt := st.chat_input("What health questions do you have?", disabled=time_limit_reached):
            # Display user message (no avatar needed for user)
            with st.chat_message("user"):
                st.markdown(prompt)
            # Add user message to chat history
            # Store None for 'small_avatar_used' for user messages
            st.session_state.messages.append({"role": "user", "content": prompt, "small_avatar_used": None})

            # Show thinking indicator within THIS column
            with st.chat_message("assistant", avatar=use_small_avatar_icon): # Use the small icon setting
                with st.spinner("Thinking..."):
                    response = get_gemini_response(
                        prompt,
                        st.session_state.messages,
                        is_adaptive=st.session_state.experiment_condition['lsm'],
                        show_avatar=st.session_state.experiment_condition['avatar'] # Pass condition info
                    )
                    st.markdown(response)

            # Add assistant response to chat history
            # Store the small avatar setting used for this message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "small_avatar_used": use_small_avatar_icon
            })

            # Rerun potentially needed to ensure layout updates correctly after adding messages
            st.rerun()


        # Check time limit again after response
        if st.session_state.start_time and (time.time() - st.session_state.start_time > 600):
            # Re-display warning if needed within this column
            # st.warning("Interaction time limit reached...") # Already displayed above/might be redundant
            pass # Input is already disabled