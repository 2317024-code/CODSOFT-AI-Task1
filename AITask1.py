def chatbot_response(user_input):
    user_input = user_input.lower()

    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"

    elif "how are you" in user_input:
        return "I'm just a bot, but I'm doing great! Thanks for asking."

    elif "who are you" in user_input or "what are you" in user_input:
        return "I am a simple rule-based chatbot created in Python."

    elif "help" in user_input:
        return "Sure! I can answer simple queries like greetings, bot info, or exit."

    elif "bye" in user_input or "exit" in user_input or "quit" in user_input:
        return "Goodbye! Have a nice day."

    else:
        return "I'm sorry, I don't understand that. Can you rephrase?"

print("Rule-Based Chatbot: Type 'bye' to exit")
while True:
    user = input("You: ")
    response = chatbot_response(user)
    print("Bot:", response)
    if "bye" in user.lower() or "exit" in user.lower() or "quit" in user.lower():
        break
