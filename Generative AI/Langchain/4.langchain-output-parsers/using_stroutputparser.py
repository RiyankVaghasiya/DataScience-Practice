from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# model = ChatGroq(
#     model="llama3-8b-8192", 
# )


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

model = ChatHuggingFace(llm=llm)

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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"black hole"})

print(result)

