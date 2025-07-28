from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


#1 method to use huggingface model locally by downloading it
pipe = pipeline("text-generation", model="openai-community/gpt2-medium",max_new_tokens=100)

model = HuggingFacePipeline(pipeline=pipe)


while True:
    user_input = input('You: ')

    if user_input == 'exit':
        break

    prompt = f"Q: {user_input}\nA:"
    result = model.invoke(prompt).strip()

    print("AI:",result)
