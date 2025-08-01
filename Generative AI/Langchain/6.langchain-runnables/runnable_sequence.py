from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template="write a joke about the {topic}",
    input_variables=['topic']
)

model = ChatGroq(
    model="llama3-8b-8192",  
    temperature=0
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="explain the following joke - {text}",
    input_variables='text'
)

chain = RunnableSequence(prompt,model,parser,prompt2,model,parser)

result = chain.invoke({'topic':"AI"})

print(result)

