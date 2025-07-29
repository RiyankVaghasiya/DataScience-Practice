from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os


hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

# model = ChatGroq(
#     model="llama3-8b-8192", 
# )

#1 first prompt we will enter for detailed prompt
template1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)


#2 second prompt we will enter for summary of that detailed prompt
template2 = PromptTemplate(
    template='write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':"black hole"})

result = model.invoke(prompt1)

prompt2 = template2.invoke(result.content)

final_summary = model.invoke(prompt2)

print(final_summary.content)

