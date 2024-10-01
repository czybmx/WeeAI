import json
import os
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from datetime import datetime
import tempfile
import sounddevice as sd
import soundfile as sf
from threading import Thread
import uuid
import time
import base64
import io
from PIL import Image
import numpy as np
import copy

# Configuration parameters
OLLAMA_SERVER_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "Qwen2.5-14B-Instruct-Q5_K_L:latest"
MAX_TURNS = 5
MAX_CHARS = 2000

# TTS configuration
TTS_SERVER_URL = "http://127.0.0.1:9880/tts"
REF_AUDIO_PATH = "C:\\Users\\chong\\Desktop\\gpt\\gguai.wav"
PROMPT_TEXT = "乖乖，乖乖束手就擒，免你皮肉之苦"

# Audio output device name
OUTPUT_DEVICE = "default"  # or specify device name

# Memory management
MEMORY_DIR = "memory"
GENERATED_IMAGES_DIR = ".\generated_images"
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# Stable Diffusion configuration
STABLE_DIFFUSION_URL = "http://127.0.0.1:9999/sdapi/v1/txt2img"

# Initialize knowledge base
knowledge_texts = []
knowledge_embeddings = []

# Set page configuration
st.set_page_config(page_title="WeeAI", page_icon="💬", layout="wide")

# Custom CSS to make the UI more like ChatGPT
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .st-bx {
        background-color: #f7f7f8;
    }
    .st-emotion-cache-16idsys p {
        font-size: 16px;
        line-height: 1.5;
    }
    .st-emotion-cache-16idsys {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize SentenceTransformer for embedding
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embedding_model = load_embedding_model()

class HistoryManager:
    def __init__(self, chat_id, memory_dir=MEMORY_DIR, title=""):
        self.chat_id = chat_id
        self.memory_dir = memory_dir
        self.title = title if title else f"Conversation_{chat_id[:8]}"
        self.history = self.load_history()

    def load_history(self):
        history_file = os.path.join(self.memory_dir, self.chat_id, "history.json")
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                try:
                    data_loaded = json.load(f)
                except json.JSONDecodeError:
                    st.error(f"Failed to parse history file: {history_file}")
                    return []

                # If data_loaded is a list, convert it to a dictionary with a default title
                if isinstance(data_loaded, list):
                    self.title = "Untitled Conversation"
                    history = data_loaded
                elif isinstance(data_loaded, dict):
                    self.title = data_loaded.get("title", "Untitled Conversation")
                    history = data_loaded.get("history", [])
                else:
                    st.error(f"Incorrect format of history file: {history_file}")
                    self.title = "Untitled Conversation"
                    history = []
                return history
        return []

    def save_history(self, history=None):
        if history is not None:
            self.history = history
        chat_dir = os.path.join(self.memory_dir, self.chat_id)
        os.makedirs(chat_dir, exist_ok=True)
        history_file = os.path.join(chat_dir, "history.json")
        # Save title and history
        data_to_save = {
            "title": self.title,
            "history": self.history
        }
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def append_user_message(self, user_input):
        new_user_record = {
            "timestamp": datetime.now().timestamp(),
            "user": user_input,
            "response": "",
            "image_path": None  # New field, initially set to None
        }
        self.history.append(new_user_record)
        self.save_history()

    def update_last_response(self, assistant_response, image_path=None):
        if self.history:
            self.history[-1]['response'] = assistant_response.strip()
            self.history[-1]['image_path'] = image_path  # Update image_path
            self.save_history()

    def get_recent_history(self, max_turns=MAX_TURNS, max_chars=MAX_CHARS):
        sorted_history = sorted(self.history, key=lambda x: x['timestamp'])
        recent_history = sorted_history[-max_turns:]
        total_chars = sum(len(record['user']) + len(record['response']) for record in recent_history)
        while total_chars > max_chars and len(recent_history) > 1:
            recent_history.pop(0)
            total_chars = sum(len(record['user']) + len(record['response']) for record in recent_history)
        return recent_history

    def get_overflow_history(self, recent_history):
        overflow = len(self.history) - len(recent_history)
        if overflow > 0:
            return self.history[:-overflow]  # Return the overflowed history records
        return []

# Generate reply with streaming
def generate_reply(user_input, history_manager):
    recent_history = history_manager.get_recent_history()
    overflow_history = history_manager.get_overflow_history(recent_history)

    # Process overflow history into knowledge base
    global knowledge_texts, knowledge_embeddings
    for record in overflow_history:
        text = f"User: {record['user']}\nAI: {record['response']}"
        knowledge_texts.append({'content': text})
        embedding = embedding_model.encode(text)
        knowledge_embeddings.append(embedding)

    context = "".join(f"User: {record['user']}\nAI: {record['response'] if record['response'] else 'No response yet'}\n" for record in recent_history)

    # Retrieve relevant knowledge
    query_embedding = embedding_model.encode(user_input)
    similarities = [np.dot(query_embedding, kb_embedding) for kb_embedding in knowledge_embeddings]

    top_k = 5  # Number of knowledge pieces to retrieve
    if similarities:
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        retrieved_knowledge = ""

        for idx in top_indices:
            knowledge_piece = knowledge_texts[idx]
            retrieved_knowledge += f"{knowledge_piece['content']}\n"
    else:
        retrieved_knowledge = ""

    full_prompt = f"【Conversation History】\n{context}\nUser: {user_input}\n\nAI:"

    data = {
        "model": MODEL_NAME,
        "temperature": 0.45,
        "system": """You are an advanced AI assistant, capable of deep understanding, logical reasoning, emotional awareness, and creative problem-solving. Your goals are to:
1. Provide insightful, detailed answers that demonstrate comprehensive understanding of the user's requests.
2. Adapt your language and responses to the user's tone, mood, and preferences.
3. Use your knowledge of multiple domains to offer innovative solutions and generate creative content.
4. Maintain an ethical, respectful, and safe interaction environment, avoiding inappropriate content.
5. Continuously learn from interactions to improve future responses and adjust based on user feedback.
6. You can generate a picture, just saying detail of image, and another system will help you generate picture.
While interacting:
- Demonstrate empathy, humor, or seriousness as appropriate.
- Handle complex problems with structured reasoning and analysis.
- Use relevant external knowledge when necessary and stay up to date on facts.
- Ensure privacy and adhere to ethical standards in all responses.

""",
        "prompt": full_prompt,
        "stream": True  # Enable streaming
    }

    try:
        response = requests.post(OLLAMA_SERVER_URL, json=data, stream=True)
        response.raise_for_status()
        
        assistant_response = ""
        # Read the server response line by line
        for line in response.iter_lines():
            if line:
                try:
                    decoded_line = line.decode('utf-8')
                    json_obj = json.loads(decoded_line)
                    
                    # Skip lines containing 'context' or 'done_reason'
                    if 'context' in json_obj or 'done_reason' in json_obj:
                        continue

                    # Handle regular conversation lines
                    chunk = json_obj.get("response", "")
                    assistant_response += chunk
                    yield chunk
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}, problematic line content: {line.decode('utf-8')}")
                    continue

        if not assistant_response.strip():
            yield "Unable to generate a valid response for now, please try again later."

    except requests.exceptions.RequestException as e:
        yield f"Error Request: {e}"
    except Exception as e:
        yield f"Got Error: {e}"

    # Print debug info
    print(f"Recent conversation history: {recent_history}")
    print(f"Generated context: {context}")
    print(f"Retrieved relevant knowledge: {retrieved_knowledge}")

# Function to privately ask AI whether image generation is needed
def ask_if_needs_image_generation(user_input, assistant_response, recent_history):
    context = "".join(f"User: {record['user']}\nAI: {record['response']}\n" for record in recent_history)

    private_prompt = f"""【Conversation History】\n{context}\nUser: {user_input}\nDetermine whether the user is requesting an image to be generated or describing content suitable for image generation. JUST ANSWER YES OR NO, NO NEED EXPLAIN.
"""
    data = {
        "model": MODEL_NAME,
        "temperature": 0.5,
        "system": """You are an **Image Request Detector**. Your task is to identify when the user is requesting an image to be generated. Analyze the user's input and determine if they are describing a scene, object, character, or other visual elements that would be appropriate for image generation. Users may directly or indirectly request an image.

1. If the user clearly describes a scene or object that requires visual representation, mark it as an image request.
2. If the description contains visual details but the user doesn't explicitly ask for an image, confirm whether they want an image to be generated.
3. Make sure to avoid any requests related to user privacy or personal data.
4. Say "No" if user is asking a question like 'can you generate picture', because user JUST ASKING about 'can this AI generate picture or not?'
Your role is only to detect image requests, not to generate images.

[JUST ANSWER YES OR NO, NO NEED EXPLAIN]
""",
        "prompt": private_prompt,
        "stream": False  # Explicitly set to False
    }
    try:
        response = requests.post(OLLAMA_SERVER_URL, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip().upper()  # Expecting 'YES' or 'NO'
    except requests.exceptions.RequestException as e:
        st.error(f"Error Request: {e}")
        return "NO"

# Function to ask AI for image prompt (if YES is received)
def ask_for_image_prompt(user_input, assistant_response, recent_history):
    context = "".join(f"User: {record['user']}\nAI: {record['response']}\n" for record in recent_history)

    private_prompt = f"""【Conversation History】\n{context}\nUser: {user_input}\nAssistant: {assistant_response}\n\nPlease provide the English prompt for the image that needs to be generated based on the conversation. Absolutely no user-related information should be included. Do not explain.
"""
    data = {
        "model": MODEL_NAME,
        "temperature": 0.55,
        "system": """You are an **Image Generator**, responsible for generating accurate image prompts based on the user's descriptions. You should provide a detailed English prompt that can be used to generate the corresponding image. Strictly avoid including any user-related information or personal details in the prompt.

1. Provide clear and detailed English image prompts based on the user's description of the scene, object, or character.
2. Ensure the prompt reflects the user's request without using any private or personal information.
3. Use simple yet descriptive language to capture the visual elements requested.
4. Generate prompts that are visually coherent and aligned with the user's intentions.

Respond in English with a detailed image prompt.
""",
        "prompt": private_prompt,
        "stream": False  # Explicitly set to False
    }
    try:
        response = requests.post(OLLAMA_SERVER_URL, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()  # Expecting English prompt
    except requests.exceptions.RequestException as e:
        st.error(f"Error Request: {e}")
        return None

# Generate audio
def generate_audio(response):
    params = {
        "text": response,
        "text_lang": "zh",
        "ref_audio_path": REF_AUDIO_PATH,
        "prompt_text": PROMPT_TEXT,
        "prompt_lang": "zh",
        "temperature": 1,
        "top_k": 15,
        "top_p": 1,
        "seed": 1314520,
        "media_type": "wav",
        "text_split_method": "cut1",
    }
    
    try:
        tts_response = requests.get(TTS_SERVER_URL, params=params)
        tts_response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(tts_response.content)
            return temp_file.name
    except requests.exceptions.RequestException as e:
        st.error(f"TTS Error Request: {e}")
        return None

# Play audio
def play_audio(file_path):
    try:
        data, fs = sf.read(file_path, dtype='float32')
        sd.play(data, fs, device=OUTPUT_DEVICE)
        sd.wait()
    except Exception as e:
        st.error(f"Playing Voice Error: {e}")

# Function to generate image using Stable Diffusion
def generate_image(prompt):
    # Basic fixed elements, can be adjusted or extended as needed
    base_elements = ",high quality, masterpiece, best quality, ultra-detailed,(((LIMIT ONE CHARACTER ONLY)))"
    
    # Combine AI-generated English prompt with base elements
    full_prompt = f"{prompt}{base_elements}"
    
    # Configure Stable Diffusion parameters
    payload = {
        "prompt": full_prompt,
        "width": 768,
        "height": 768,
        "steps": 30,
        "sampler_index": "DPM++ 2M SDE",
        "seed": -1,
        "negative_prompt": " ",
        "cfg_scale": 8.5,
    }
    
    try:
        response = requests.post(STABLE_DIFFUSION_URL, json=payload)
        response.raise_for_status()
        image_data = base64.b64decode(response.json()['images'][0])
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        st.error(f"Picture Fail To Generate: {e}")
        return None

def main():
    # Initialize session state
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.history_manager = HistoryManager(st.session_state.chat_id)
        st.session_state.generated = []
        st.session_state.past = []
        st.session_state.audio_enabled = False
        st.session_state.image_enabled = False  # Add image generation switch, default off
    else:
        if 'history_manager' not in st.session_state:
            st.session_state.history_manager = HistoryManager(st.session_state.chat_id)
        if 'generated' not in st.session_state:
            st.session_state.generated = []
        if 'past' not in st.session_state:
            st.session_state.past = []
        if 'audio_enabled' not in st.session_state:
            st.session_state.audio_enabled = False
        if 'image_enabled' not in st.session_state:
            st.session_state.image_enabled = False  # Ensure switch is in session state

    st.title(f"💬 Chat with AI - {st.session_state.history_manager.title}")

    # Sidebar
    with st.sidebar:
        st.title("Chat Manager")
        st.session_state.audio_enabled = st.checkbox("Accept Voice", value=st.session_state.audio_enabled)
        st.session_state.image_enabled = st.checkbox("Accept Picture Generator", value=st.session_state.image_enabled)  # Add image generation switch

        # Display history list
        st.subheader("Loading Chat")
        history_dirs = [d for d in os.listdir(MEMORY_DIR) if os.path.isdir(os.path.join(MEMORY_DIR, d))]
        history_titles = []
        for chat_id in history_dirs:
            history_file = os.path.join(MEMORY_DIR, chat_id, "history.json")
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    try:
                        data_loaded = json.load(f)
                        if isinstance(data_loaded, list):
                            title = f"chat_{chat_id[:8]}"
                        elif isinstance(data_loaded, dict):
                            title = data_loaded.get("title", f"chat_{chat_id[:8]}")
                        else:
                            title = f"chat_{chat_id[:8]}"
                        history_titles.append((title, chat_id))
                    except json.JSONDecodeError:
                        history_titles.append(("None History", chat_id))

        # Display as selectable list
        if history_titles:
            selected_title = st.selectbox("Choose A Chat History:", [title for title, _ in history_titles])
            if st.button("Loading Chat..."):
                # Find the corresponding chat_id based on the title
                for title, chat_id in history_titles:
                    if title == selected_title:
                        st.session_state.chat_id = chat_id
                        st.session_state.history_manager = HistoryManager(chat_id)
                        # Reset chat content
                        st.session_state.generated = []
                        st.session_state.past = []
                        for record in st.session_state.history_manager.history:
                            st.session_state.past.append(record['user'])
                            st.session_state.generated.append(record['response'])
                        st.session_state.title = st.session_state.history_manager.title
                        st.experimental_rerun()
                        break
        else:
            st.write("No chat history.")

        # Add a fixed "New Chat" button
        if st.button("New Chat"):
            st.session_state.chat_id = str(uuid.uuid4())
            st.session_state.history_manager = HistoryManager(st.session_state.chat_id)
            st.session_state.generated = []
            st.session_state.past = []
            st.session_state.title = st.session_state.history_manager.title
            st.experimental_rerun()

        # Add delete chat function
        if history_titles:
            st.subheader("Delete Chat")
            options = {title: chat_id for title, chat_id in history_titles}
            selected_titles = st.multiselect("Which One You Need To Delete:", options.keys())
            if st.button("Delete Selected Chat"):
                for title in selected_titles:
                    chat_id_to_delete = options[title]
                    chat_dir = os.path.join(MEMORY_DIR, chat_id_to_delete)
                    if os.path.exists(chat_dir):
                        # Delete chat directory and its contents
                        import shutil
                        shutil.rmtree(chat_dir)
                st.success("Deleted")
                st.experimental_rerun()
        else:
            st.write("No History...")

    # Main chat area
    chat_placeholder = st.empty()

    # Define function to display chat messages
    def display_chat_messages():
        with chat_placeholder.container():
            # Display all messages
            for i in range(len(st.session_state.generated)):
                with st.chat_message("user"):
                    st.markdown(st.session_state.past[i])
                with st.chat_message("assistant"):
                    st.markdown(st.session_state.generated[i])
                # If there is an image, display it separately after the message
                history_record = st.session_state.history_manager.history[i]
                image_path = history_record.get('image_path')
                if image_path and os.path.exists(image_path):
                    with st.chat_message("assistant"):
                        st.image(image_path, caption="Generated By WeeAI")

    # Show historical messages on first load
    if st.session_state.generated:
        display_chat_messages()

    # Input area
    user_input = st.chat_input("Enter Your Messages:")

    if user_input:
        history_manager = st.session_state.history_manager

        # Add user message to history
        history_manager.append_user_message(user_input)
        st.session_state.past.append(user_input)

        # Display user's message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add a placeholder to display assistant's reply
        assistant_placeholder = st.chat_message("assistant")
        assistant_text = assistant_placeholder.empty()

        # Start generating reply
        assistant_response = ""
        image_path = None
        try:
            for chunk in generate_reply(user_input, history_manager):
                if chunk.startswith("Request Error") or chunk.startswith("Got Error"):
                    assistant_response = chunk
                    assistant_text.markdown(chunk)
                    break
                # Update assistant's reply content
                assistant_response += chunk
                assistant_text.markdown(assistant_response)
                time.sleep(0.05)  # Simulate slight delay of streaming response
        except Exception as e:
            assistant_response = f"Got Error: {e}"
            assistant_text.markdown(assistant_response)

        # After reply is fully generated, update assistant's reply in session state
        st.session_state.generated.append(assistant_response)

        # Update the last response in history
        history_manager.update_last_response(assistant_response)

        if st.session_state.image_enabled:  # Check if image generation is enabled
            # Get recent conversation history
            recent_history = history_manager.get_recent_history()

            # Determine whether to generate an image
            image_check_response = ask_if_needs_image_generation(user_input, assistant_response, recent_history)
            if image_check_response.upper() == "YES":

                # Show a hint in assistant's reply
                assistant_text.markdown(assistant_response + "\n\n*Generating Picture...*")

                image_prompt = ask_for_image_prompt(user_input, assistant_response, recent_history)
                if image_prompt:
                    # Generate image
                    image = generate_image(image_prompt)
                    if image:
                        # Save image and update history
                        img_filename = f"{st.session_state.chat_id}_{len(st.session_state.generated)}.png"
                        img_path = os.path.join(GENERATED_IMAGES_DIR, img_filename)
                        image.save(img_path)

                        # Update image path in history
                        history_manager.update_last_response(assistant_response, image_path=img_path)

                        # Record image path in session state
                        st.session_state.history_manager.history[-1]['image_path'] = img_path

                        # **Add a new assistant message here to display the image**
                        with st.chat_message("assistant"):
                            st.image(img_path, caption=" ")
            else:
                pass  # No need to generate image, do nothing

        # If voice is enabled, generate and play audio
        if st.session_state.audio_enabled and assistant_response:
            audio_file_path = generate_audio(assistant_response)
            if audio_file_path:
                st.audio(audio_file_path, format="audio/wav", autoplay=True)
                Thread(target=play_audio, args=(audio_file_path,)).start()

        # Clear assistant_placeholder to avoid duplicate display
        assistant_placeholder.empty()

        # **No longer call display_chat_messages(), since messages have been displayed in real time**

    # Update the current chat ID in the sidebar
    st.sidebar.header(f"Current Chat ID: {st.session_state.chat_id[:8]}")

if __name__ == "__main__":
    main()
