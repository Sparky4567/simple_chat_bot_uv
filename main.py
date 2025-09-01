from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
import warnings
from sqlalchemy.exc import SAWarning
from icecream import ic

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

# Create the chatbot
chatbot = ChatBot("ALIS v.1.0")

# 1. Train with the built-in English corpus
#corpus_trainer = ChatterBotCorpusTrainer(chatbot)
#corpus_trainer.train("chatterbot.corpus.english")

# 2. Keep track of conversation history for retraining
conversation_history = []
history_trainer = ListTrainer(chatbot)
print("SimpleBot is ready. Type 'quit' to exit, 'retrain' to retrain on past chats.\n\n")

def starter_function():
    user_input = str(input("You: ")).strip()
    match user_input:
        case "quit":
            quit()
        case "retrain":
            if(conversation_history):
                print("Retraining on past conversation...")
                history_trainer.train(conversation_history)
                starter_function()
            else:
                print("No conversation history yet. Check if the variable is defined.\n\n")
                starter_function()
        case _:
            conversation_history.append(user_input)
            print("User input - {} - was appended to history".format(user_input))
            try:
                bot_response = chatbot.get_response(user_input)
                if not str(bot_response).strip():
                    bot_response = "I'm not sure what to say to that."
                    ic("Bot: {}".format(bot_response))
                else:
                    ic("Bot: {}".format(bot_response))
                conversation_history.append(str(bot_response).strip())
                print("Bot answer - {} - was appended to history".format(user_input))
                starter_function()
            except Exception as e:
                print("Error: {}".format(e))
                bot_response = "Something went wrong."
                starter_function()
            except KeyboardInterrupt:
                print("Quiting...")
                quit()

starter_function()