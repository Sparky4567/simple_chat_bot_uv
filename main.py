from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
import warnings
from sqlalchemy.exc import SAWarning
from icecream import ic
from difflib import SequenceMatcher

# --- Config ---
USE_SIMILARITY_SCORING = True  # toggle similarity scoring on/off
SIMILARITY_THRESHOLD = 0.3     # tweak sensitivity if enabled

# --- Suppress noisy SAWarning ---
# This warning happens because Bot creates Statement objects
# outside the current SQLAlchemy session, then touches relationships.
# It's harmless in practice (Bot handles persistence on its own),
# but noisy. We filter it here to keep logs clean without affecting
# actual DB operations.
warnings.filterwarnings(
    "ignore",
    category=SAWarning,
    message=".*Object of type <Statement> not in session.*"
)

# --- Create the chatbot ---
chatbot = ChatBot("ALIS v.1.0")

# 1. Train with the built-in English corpus (optional)
# corpus_trainer = ChatterBotCorpusTrainer(chatbot)
# corpus_trainer.train("chatterbot.corpus.english")

# 2. Keep track of conversation history for retraining
conversation_history = []
history_trainer = ListTrainer(chatbot)

print("SimpleBot is ready. Type 'quit' to exit, 'retrain' to retrain on past chats.\n\n")

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

# --- Main loop ---
def starter_function():
    user_input = str(input("You: ")).strip()
    match user_input:
        case "quit":
            quit()
        case "retrain":
            if conversation_history:
                print("Retraining on past conversation...")
                history_trainer.train(conversation_history)
                starter_function()
            else:
                print("No conversation history yet. Check if the variable is defined.\n\n")
                starter_function()
        case _:
            conversation_history.append(user_input)
            print(f"User input - {user_input} - was appended to history")

            try:
                bot_response = chatbot.get_response(user_input)

                if USE_SIMILARITY_SCORING:
                    best_match, score = get_best_match(user_input, chatbot)
                    ic(f"Best match: {best_match}, Score: {score:.2f}")
                    if score < SIMILARITY_THRESHOLD:
                        bot_response = "I'm not sure what to say to that."

                ic(f"Bot: {bot_response}")
                conversation_history.append(str(bot_response).strip())
                print(f"Bot answer - {bot_response} - was appended to history")
                starter_function()

            except Exception as e:
                print(f"Error: {e}")
                bot_response = "Something went wrong."
                starter_function()
            except KeyboardInterrupt:
                print("Quitting...")
                quit()

starter_function()
