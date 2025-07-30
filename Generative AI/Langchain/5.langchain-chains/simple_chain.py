from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="llama3-8b-8192",  
    # temperature=0
)

prompt = PromptTemplate(
    template="generate 5 intresting facts about the {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic" : "black hole"})

print(result)

chain.get_graph().print_ascii()