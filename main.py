from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer

# Create the chatbot
chatbot = ChatBot("SimpleBot")

# 1. Train with the built-in English corpus
#corpus_trainer = ChatterBotCorpusTrainer(chatbot)
#corpus_trainer.train("chatterbot.corpus.english")

# 2. Keep track of conversation history for retraining
conversation_history = []
history_trainer = ListTrainer(chatbot)

print("SimpleBot is ready. Type 'quit' to exit, 'retrain' to retrain on past chats.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    elif user_input.lower() == "retrain":
        if conversation_history:
            print("Retraining on past conversation...")
            history_trainer.train(conversation_history)
        else:
            print("No conversation history yet.")
        continue

    conversation_history.append(user_input)

    try:
        bot_response = chatbot.get_response(user_input)
        if not str(bot_response).strip():
            bot_response = "I’m not sure what to say to that."
    except Exception:
        bot_response = "Hmm, something went wrong."

    print(f"Bot: {bot_response}")
    conversation_history.append(str(bot_response))
