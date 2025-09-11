from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import warnings
from sqlalchemy.exc import SAWarning
from icecream import ic
from difflib import SequenceMatcher
from piper.voice import PiperVoice
import os
import wave
import sounddevice as sd
import soundfile as sf
import time
# --- Config ---
USE_SIMILARITY_SCORING = True

# ollama

USE_LOCAL_LLM = False
DEFAULT_LLM_MODEL = "tinyllama:latest"
# Initialize the Ollama LLM
llm = OllamaLLM(model=DEFAULT_LLM_MODEL)

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template="Q: {question}\nA:"
)


SPHINX=False
SPEAK_BACK=True



SIMILARITY_THRESHOLD = 0.7



BOT_NAME = "ALIS v.1.0"
VOICE_MODEL_PATH = os.path.join(os.getcwd(), "semane", "en_GB-semaine-medium.onnx")

# --- Suppress noisy SAWarning ---
warnings.filterwarnings(
    "ignore",
    category=SAWarning,
    message=".*Object of type <Statement> not in session.*"
)

# --- Initialize chatbot ---
chatbot = ChatBot(BOT_NAME)
conversation_history = []
history_trainer = ListTrainer(chatbot)

# --- Initialize Piper TTS ---
voice = PiperVoice.load(VOICE_MODEL_PATH)

import sounddevice as sd
from piper.voice import PiperVoice
from pocketsphinx import LiveSpeech


def get_response_from_llm(passed_prompt):
    try:
        formatted_prompt = prompt.format(question=passed_prompt)
        text=""
        for chunk in llm.stream(formatted_prompt):
            print(chunk, end='', flush=True)
            text+=chunk
        print("\n\n")
        text+="\n\n"
        return text
    except Exception as e:
        print(f"LLM Error: {e}")
        get_response_from_llm(passed_prompt)

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


def chunk_print(text, chunk_size=10, delay=0.2):
    for i in range(0, len(text), chunk_size):
        print(text[i:i+chunk_size], end='', flush=True)
        time.sleep(delay)
    print() 


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



# --- Similarity helpers ---
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_best_match(user_input, chatbot):
    responses = [str(statement.text) for statement in chatbot.storage.filter()]
    best_match, best_score = None, 0
    for response in responses:
        score = similarity(user_input, response)
        if score > best_score:
            best_match, best_score = response, score
    return best_match, best_score

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
# --- Main loop ---
def starter_function():
    try:
        user_input = get_user_input()
        match user_input:
            case "quit":
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
                print(f"User input - {user_input} - was appended to history")

                try:
                    if USE_LOCAL_LLM:
                        bot_response = get_response_from_llm(user_input)
                    else:
                        bot_response = chatbot.get_response(user_input)
                        chunk_print(f"Bot: {bot_response}")

                        if USE_SIMILARITY_SCORING:
                            best_match, score = get_best_match(user_input, chatbot)
                            ic(f"Best match: {best_match}, Score: {score:.2f}")
                            if score < SIMILARITY_THRESHOLD:
                                bot_response = "I'm not sure what to say to that."

                    conversation_history.append(str(bot_response).strip())
                    print(f"Bot answer - {bot_response} - appended to history")

                    # --- Speak the bot response ---
                    speak(str(bot_response))

                    starter_function()

                except Exception as e:
                    print(f"Error: {e}")
                    bot_response = "Something went wrong."
                    starter_function()
                except KeyboardInterrupt:
                    print("Quitting...")
                    quit()
    except Exception as e:
        print(f"Error: {e}")
        bot_response = "Something went wrong."
        starter_function()
    except KeyboardInterrupt:
        print("Quitting...")
        quit()

print(f"{BOT_NAME} is ready. Type 'quit' to exit, 'retrain' to retrain on past chats.\n\n")
starter_function()
