from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize the Groq LLM (choose any available model)
llm = ChatGroq(
    model="llama3-8b-8192",  # Alternatives: gemma-7b-it, mixtral-8x7b-32768
    temperature=0
)

# Initialize chat history with a system message
chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))

    # Invoke model with entire history
    response = llm.invoke(chat_history)

    # Print and store AI response
    print("AI:", response.content)
    chat_history.append(AIMessage(content=response.content))
