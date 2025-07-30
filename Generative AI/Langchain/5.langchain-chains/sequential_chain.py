from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()


model = ChatGroq(
    model="llama3-8b-8192",  
    # temperature=0
)

template1 = PromptTemplate(
    template="write a detailed report on the {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="write the summary on the: \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)

chain.get_graph().print_ascii()