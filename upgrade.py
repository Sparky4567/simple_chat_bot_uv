from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import warnings
from sqlalchemy.exc import SAWarning
from icecream import ic
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import json
import os
import time
import wave
import sounddevice as sd
import soundfile as sf
from piper.voice import PiperVoice


# --- Config ---
USE_SIMILARITY_SCORING = True
SIMILARITY_THRESHOLD = 0.8
CONTEXT_WINDOW = 3
MEMORY_FILE = "conversation_memory.json"
SPHINX=False
SPEAK_BACK=True
# --- Suppress noisy SAWarning ---
warnings.filterwarnings(
    "ignore",
    category=SAWarning,
    message=".*Object of type <Statement> not in session.*"
)

# --- Chatbot setup ---
chatbot = ChatBot("ALIS v.3.0")
history_trainer = ListTrainer(chatbot)
# --- Initialize Piper TTS ---
VOICE_MODEL_PATH = os.path.join(os.getcwd(), "semane", "en_GB-semaine-medium.onnx")
voice = PiperVoice.load(VOICE_MODEL_PATH)
# --- Long-term memory helpers ---
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def listen_once(silence_threshold=10):
    """
    Listen for a single phrase and return it after detecting silence.
    
    :param silence_threshold: seconds of silence required to close phrase
    """
    buffer = []
    last_speech_time = None
    print("Speak now\n\n")
    for phrase in LiveSpeech():
        text = str(phrase).strip()

        if text:
            if not buffer:
                print("[Speech started]")
            buffer.append(text)
            last_speech_time = time.time()
        else:
            if buffer and last_speech_time:
                elapsed = time.time() - last_speech_time
                if elapsed > silence_threshold:
                    print("[Speech ended]")
                    return " ".join(buffer)
                
def speak(text):
    if SPEAK_BACK:
        file_name = "test.wav"
        try:
            with wave.open(file_name, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            data, samplerate = sf.read(file_name, dtype='float32')

            # Play it
            sd.play(data, samplerate=samplerate)
            sd.wait()
        except Exception as e:
            print(f"Speech error: {e}")

def save_memory(history):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

conversation_history = load_memory()

# --- Semantic model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Similarity helpers ---
def semantic_similarity(input_text, candidate_texts):
    if not candidate_texts:
        return None, 0.0
    input_emb = model.encode(input_text, convert_to_tensor=True)
    candidate_embs = model.encode(candidate_texts, convert_to_tensor=True)
    scores = util.cos_sim(input_emb, candidate_embs)
    best_idx = scores.argmax()
    return candidate_texts[best_idx], float(scores[0][best_idx])

def get_context_window(history, window_size=CONTEXT_WINDOW):
    return " ".join(history[-window_size*2:])  # user + bot messages


def get_user_input():
    try:
        if SPHINX:
            user_input = listen_once()
        else:
            user_input = str(input("You: ")).strip()
        return str(user_input).strip()
    except Exception as e:
        print("Exception: {e}".format(e))
        starter_function()

# --- Fallback rules ---
def fallback_rules(user_input):
    if any(greet in user_input.lower() for greet in ["hi", "hello", "hey"]):
        return "Hello! How are you today?"
    if "time" in user_input.lower():
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}"
    return "I'm not sure what to say to that."

# --- Main loop ---
def starter_function():
    user_input = get_user_input()
    match user_input:
        case "quit":
            save_memory(conversation_history)
            quit()
        case "retrain":
            if conversation_history:
                print("Retraining on past conversation...")
                history_trainer.train(conversation_history)
                starter_function()
            else:
                print("No conversation history yet.\n")
                starter_function()
        case _:
            conversation_history.append(user_input)
            print(f"User input - {user_input} - appended to history")

            try:
                bot_response = chatbot.get_response(user_input)
                source = "ChatterBot"

                if USE_SIMILARITY_SCORING:
                    context_text = get_context_window(conversation_history)
                    candidate_texts = [str(statement.text) for statement in chatbot.storage.filter()]
                    best_match, score = semantic_similarity(context_text + " " + user_input, candidate_texts)
                    
                    if score >= SIMILARITY_THRESHOLD:
                        bot_response = best_match
                        source = f"Semantic Match (score={score:.2f})"
                    else:
                        bot_response = fallback_rules(user_input)
                        source = "Fallback Rule"

                ic(f"Bot: {bot_response} | Source: {source}")
                conversation_history.append(str(bot_response).strip())
                save_memory(conversation_history)
                print(f"Bot answer - {bot_response} - appended to history")
                 # --- Speak the bot response ---
                speak(str(bot_response))
                starter_function()

            except Exception as e:
                print(f"Error: {e}")
                bot_response = "Something went wrong."
                starter_function()
            except KeyboardInterrupt:
                save_memory(conversation_history)
                print("Quitting...")
                quit()

starter_function()
